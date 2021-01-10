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
import json
import shutil




def train_image_model(userName,projectName,projectType):

    S3_BUCKET_OUTPUT = 'gauravp-eva4-capstone-models'
    # Find number of users
    s3 = boto3.client('s3',aws_access_key_id='aws_access_key_id',aws_secret_access_key='aws_secret_access_key')

    print('Delete model files corresponding to current session')
    
    savedModelName = f'{userName}_{projectName}_{projectType}.pt'
    print(f'Deleting {savedModelName}')
    s3.delete_object(Bucket=S3_BUCKET_OUTPUT,Key=savedModelName)

    model_info_file_name = f'{userName}_{projectName}_{projectType}.json'
    print(f'Deleting {model_info_file_name}')
    s3.delete_object(Bucket=S3_BUCKET_OUTPUT,Key=model_info_file_name)


    print('Preparing train val splits')
    input_folder = f'./user_data/{userName}/{projectName}/{projectType}/train_data'
    print("Train Image Folder:",input_folder)
    dataset_folder = 'data'

    splitfolders.ratio(input_folder, output=dataset_folder, seed=1337, ratio=(0.7, 0.3))

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
    class_names = dataset_train.classes
    print(f'Number of Classes = {num_classes}')
    print(f'Class Names: {class_names}')


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
    num_epochs = 10
    dataset_sizes = {'train':len(dataset_train),'val':len(dataset_valid)}

    since = time.time()
    trainLossLog = []
    testLossLog = []
    trainAccLog = []
    testAccLog = []
    history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_train_acc = 0.0

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

            if phase == 'train' and epoch_acc > best_train_acc:
                best_train_acc = epoch_acc
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best Training Acc: {:4f}'.format(best_train_acc))
    print('Best val Acc: {:4f}'.format(best_acc))
    

    # load best model weights
    model.load_state_dict(best_model_wts)

    traced_model = torch.jit.trace(model,torch.randn(1,3,224,224))
    savedModelName = f'{userName}_{projectName}_{projectType}.pt'
    print("Saved Model Name",savedModelName)
    traced_model.save(savedModelName)

    # prepare model information file
    model_info = {}
    model_info['numClasses'] = num_classes
    model_info['classNames'] = class_names
    model_info['modelName'] = savedModelName
    model_info['userName'] = userName
    model_info['projectName'] = projectName
    model_info['bestTestAcc'] = best_acc.item()
    model_info['bestTrainAcc'] = best_train_acc.item()
    print(model_info)

    model_info_file_name = f'{userName}_{projectName}_{projectType}.json'
    with open(model_info_file_name, "w") as outfile:
                json.dump(model_info, outfile)

    print('Saving model info and model to s3')
#    S3_BUCKET_OUTPUT = 'gauravp-eva4-capstone-models'
    # Find number of users
#    s3 = boto3.client('s3',aws_access_key_id='aws_access_key_id',aws_secret_access_key='aws_secret_access_key')

    s3.upload_file(model_info_file_name, S3_BUCKET_OUTPUT, model_info_file_name,)
    s3.upload_file(savedModelName, S3_BUCKET_OUTPUT, savedModelName)
    print("Done!!!")
