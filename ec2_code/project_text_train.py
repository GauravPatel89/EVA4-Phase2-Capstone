# Import Libraries
import io
import random
import os, pickle
import torch, torchtext
from torchtext import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import json
import boto3
import csv

class classifier(nn.Module):

    # Define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer
        self.encoder = nn.LSTM(embedding_dim,
                               hidden_dim,
                               num_layers=n_layers,
                               dropout=dropout,
                               batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, text_lengths):

        # text = [batch size, sent_length]
        embedded = self.embedding(text)
        # embedded = [batch size, sent_len, emb dim]

        # packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True)

        packed_output, (hidden, cell) = self.encoder(packed_embedded)
        # hidden = [batch size, num layers * num directions,hid dim]
        # cell = [batch size, num layers * num directions,hid dim]

        # Hidden = [batch size, hid dim * num directions]
        dense_outputs = self.fc(hidden)

        # Final activation function softmax
        output = F.softmax(dense_outputs[0], dim=1)

        return output




def train_text_model(userName,projectName,projectType,numEpochs=10):

    S3_BUCKET_OUTPUT = 'gauravp-eva4-capstone-models'
    # Find number of users
    s3 = boto3.client('s3',aws_access_key_id='aws_access_key_id',aws_secret_access_key='aws_secre')

    print('Delete model files corresponding to current session')
    
    savedTokenizerName = f'{userName}_{projectName}_{projectType}.pkl' 
    print(f'Deleting {savedTokenizerName}')
    s3.delete_object(Bucket=S3_BUCKET_OUTPUT,Key=savedTokenizerName)

    savedModelName = f'{userName}_{projectName}_{projectType}.pt'
    print(f'Deleting {savedModelName}')
    s3.delete_object(Bucket=S3_BUCKET_OUTPUT,Key=savedModelName)

    model_info_file_name = f'{userName}_{projectName}_{projectType}.json'
    print(f'Deleting {model_info_file_name}')
    s3.delete_object(Bucket=S3_BUCKET_OUTPUT,Key=model_info_file_name)
 
    print('Preparing train val splits')
    datasetPath = f'./user_data/{userName}/{projectName}/{projectType}/train_data'
    

    texts = []
    labels = []
    for dirName in os.listdir(datasetPath):
        dirPath = os.path.join(datasetPath,dirName)
        print(dirPath)
        #print(resizedDirPath)

        count = 0
        for fileName in os.listdir(dirPath):
            filePath = os.path.join(dirPath,fileName)
            #print(filePath)
            labelName = filePath.split('/')[-2]
            print('className: ',labelName,filePath)
            
            with open(filePath, newline='') as f:
                reader = csv.reader(f)
                row = next(reader)
                print(row)
                texts.append(row[0])
                labels.append(labelName)



    print(len(texts))
    print(len(labels))

    # Defining Fields
    # We are using spacy as a tokanizer
    dataset_text = data.Field(sequential = True, tokenize = 'spacy', batch_first =True, include_lengths=True)
    dataset_label = data.LabelField(tokenize ='spacy', is_target=True, batch_first =True, sequential =False)

    # Define names of dataset and its label
    fields = [('dataset_text', dataset_text),('dataset_label',dataset_label)]

    # We will gather data into a list
    example = [data.Example.fromlist([texts[i] ,labels[i]], fields) for i in range(len(texts))]
    
    # Define userDataset consisting of data from dataframe and fields defined by us
    userDataset = data.Dataset(example, fields)

    # split dataset into training and validation
    (train, valid) = userDataset.split(split_ratio=[0.70, 0.30])
    print((len(train), len(valid)))

    print(vars(train.examples[10]))

    # Build vacab for text data as well as text labels
    dataset_text.build_vocab(train)
    dataset_label.build_vocab(train)


    num_classes = len(dataset_label.vocab)

    print('Size of input vocab : ', len(dataset_text.vocab))
    print('Size of label vocab : ', len(dataset_label.vocab))
    print('Top 10 words appreared repeatedly :', list(dataset_text.vocab.freqs.most_common(10)))
    print('Labels : ', dataset_label.vocab.stoi)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_iterator, valid_iterator = data.BucketIterator.splits((train, valid), batch_size = 32,
                                                        sort_key = lambda x: len(x.dataset_text),
                                                        sort_within_batch=True, device = device)


    with open('tokenizer.pkl', 'wb') as tokens:
        pickle.dump(dataset_text.vocab.stoi, tokens)

    # Define hyperparameters
    size_of_vocab = len(dataset_text.vocab)
    embedding_dim = 300
    num_hidden_nodes = 100
    num_output_nodes = len(dataset_label.vocab)
    num_layers = 2
    dropout = 0.2

    # Instantiate the model
    model = classifier(size_of_vocab, embedding_dim, num_hidden_nodes, num_output_nodes, num_layers, dropout = dropout)
    print(model)


    # No. of trianable parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print(f'The model has {count_parameters(model):,} trainable parameters')


    import torch.optim as optim

    # define optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    criterion = nn.CrossEntropyLoss()


    # define metric
    def binary_accuracy(preds, y):
        # round predictions to the closest integer
        _, predictions = torch.max(preds, 1)

        correct = (predictions == y).float()
        acc = correct.sum() / len(correct)
        return acc


    # push to cuda if available
    model = model.to(device)
    criterion = criterion.to(device)

    # train loop
    def train(model, iterator, optimizer, criterion):
        # initialize every epoch
        epoch_loss = 0
        epoch_acc = 0

        # set the model in training phase
        model.train()

        for batch in iterator:
            # resets the gradients after every batch
            optimizer.zero_grad()

            # retrieve text and no. of words
            dataset_text, dataset_text_lengths = batch.dataset_text

            # convert to 1D tensor
            predictions = model(dataset_text, dataset_text_lengths).squeeze()

            # compute the loss
            loss = criterion(predictions, batch.dataset_label)

            # compute the binary accuracy
            acc = binary_accuracy(predictions, batch.dataset_label)

            # backpropage the loss and compute the gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    #evaluate loop
    def evaluate(model, iterator, criterion):
        # initialize every epoch
        epoch_loss = 0
        epoch_acc = 0

        # deactivating dropout layers
        model.eval()

        # deactivates autograd
        with torch.no_grad():
            for batch in iterator:
                # retrieve text and no. of words
                dataset_text, dataset_text_lengths = batch.dataset_text

                # convert to 1d tensor
                predictions = model(dataset_text, dataset_text_lengths).squeeze()

                # compute loss and accuracy
                loss = criterion(predictions, batch.dataset_label)
                acc = binary_accuracy(predictions, batch.dataset_label)

                # keep track of loss and accuracy
                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)


    N_EPOCHS = numEpochs
    best_valid_loss = float('inf')
    best_valid_acc = 0.0
    best_train_acc = 0.0

    for epoch in range(N_EPOCHS):

        # train the model
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)

        # evaluate the model
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), './saved_weights.pt')

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc

        if train_acc > best_train_acc:
            best_train_acc = train_acc

        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}% \n')


    
    model.load_state_dict(torch.load('./saved_weights.pt'))

    savedModelName = f'{userName}_{projectName}_{projectType}.pt'
    print("Saved Model Name",savedModelName)
    torch.save(model,savedModelName) 

    savedTokenizerName = f'{userName}_{projectName}_{projectType}.pkl'
    os.rename('./tokenizer.pkl',savedTokenizerName)
    # prepare model information file
    model_info = {}
    model_info['numClasses'] = num_classes
    model_info['classNames'] = dataset_label.vocab.itos
    model_info['modelName'] = savedModelName
    model_info['userName'] = userName
    model_info['projectName'] = projectName
    model_info['bestTestAcc'] = best_valid_acc
    model_info['bestTrainAcc'] = best_train_acc
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
    s3.upload_file(savedTokenizerName, S3_BUCKET_OUTPUT, savedTokenizerName)
    print("Done!!!")
        
