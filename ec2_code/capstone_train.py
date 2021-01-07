import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD 
import numpy as np

import torchvision
from torchvision import datasets, models, transforms

import time
import os
import copy
from PIL import Image
import splitfolders
import boto3
import os


def list_folders(s3_client, bucket_name):
	response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix='', Delimiter='/')
	for content in response.get('CommonPrefixes', []):
		yield content.get('Prefix')

# Find number of users

s3 = boto3.client('s3',aws_access_key_id='***aws_access_key_id****',aws_secret_access_key='***aws_secret_access_key***')
folder_list = list_folders(s3, 'gauravp-eva4-capstone-bucket')

print(f'Found users:{folder_list} ')
for folder in folder_list:
	print('Folder found: %s' % folder)

# Download All the files
for key in s3.list_objects(Bucket='gauravp-eva4-capstone-bucket')['Contents']:
        if key['Key'].endswith('/'):
                if not os.path.exists('./user_data/'+key['Key']):
                        os.makedirs('./user_data/'+key['Key']) 
        else:
                print('key.name',key['Key'])
                if not os.path.exists('./user_data/'+key['Key']):
                        dir_name = './user_data/'+ '/'.join(key['Key'].split('/')[:-1])
                        print('dir_name',dir_name)
                        if not os.path.exists(dir_name):
                                os.makedirs(dir_name)
                        s3.download_file('gauravp-eva4-capstone-bucket', key['Key'],'./user_data/'+key['Key'])


# Distribute data into train and test data

print('Preparing train val splits')
input_folder = './user_data/gp/mobilenet' + '/train_data'
dataset_folder = 'data'

splitfolders.ratio(input_folder, output=dataset_folder, seed=1337, ratio=(0.8, 0.2))

#Prepare data (data transforms and dataloaders)
image_transforms = {
        # Train uses data augmentation
        'train':
        transforms.Compose([
                            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                            transforms.RandomRotation(degrees=15),
                            transforms.ColorJitter(),
                            transforms.RandomHorizontalFlip(),
                            transforms.CenterCrop(size=224),  # Image net standards
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                   [0.229, 0.224, 0.225])  # Imagenet standards
                            ]),
                        # Validation does not use augmentation
        'val':
        transforms.Compose([
                            transforms.Resize(size=256),
                            transforms.CenterCrop(size=224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
        }

print('Preparing Dataloaders')
batch_size = 32

print(dataset_folder + '/train/')
dataset_train = torchvision.datasets.ImageFolder(root=(dataset_folder + '/train/'), transform=image_transforms['train'])
dataset_valid = torchvision.datasets.ImageFolder(root=(dataset_folder + '/val/'), transform=image_transforms['val'])
data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
data_loader_valid = torch.utils.data.DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=True, num_workers=4)

num_classes = len(dataset_train.classes)
print(f'Number of Classes = {num_classes}')

# Prepare the model
print('Preaparing Model for Training')
model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
    # Freeze Early layers
for param in model.parameters():
    param.requires_grad = False
    # Add custom classifier
classifierInFeatures = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Linear(classifierInFeatures, 256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, num_classes))

# prepare for training

criterion = nn.CrossEntropyLoss()   # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


print('Starting Training')
# train
device = "cpu"
num_epochs = 3
dataset_sizes = {'train':len(dataset_train),'val':len(dataset_valid)}

since = time.time()
trainLossLog = []
testLossLog = []
trainAccLog = []
testAccLog = []
history = []

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
            current_dataloader = data_loader_train
        else:
            model.eval()   # Set model to evaluate mode
            current_dataloader = data_loader_valid

        running_loss = 0.0
        running_corrects = 0
        
        # Iterate over data.
        for batch_idx, (inputs, labels) in enumerate(current_dataloader):
        #for inputs, labels in current_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        if phase == 'train':
            scheduler.step()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))

        if phase == 'train':
          trainLossLog.append(epoch_loss)
          trainAccLog.append(epoch_acc)
        else:
          testLossLog.append(epoch_loss)
          testAccLog.append(epoch_acc)

        # deep copy the model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print()

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


print('Best val Acc: {:4f}'.format(best_acc))

# load best model weights
model.load_state_dict(best_model_wts)

traced_model = torch.jit.trace(model,torch.randn(1,3,224,224))
traced_model.save('mobilenet_v2.pt')

s3.upload_file('mobilenet_v2.pt', 'gauravp-eva4-capstone-bucket', 'mobilenet_v2.pt')

