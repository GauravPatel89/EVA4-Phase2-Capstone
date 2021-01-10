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
import botocore
import os
import json
import shutil

from project_image_train import train_image_model
from project_text_train import train_text_model

S3_BUCKET_INPUT = 'gauravp-eva4-capstone-bucket'
S3_BUCKET_OUTPUT = 'gauravp-eva4-capstone-models'
# Find number of users
s3 = boto3.client('s3',aws_access_key_id='aws_access_key_id',aws_secret_access_key='aws_secret_access_key')

if os.path.exists('./user_data'):
        shutil.rmtree('./user_data')

if os.path.exists('./data'):
        shutil.rmtree('./data')


config_file_key = 'config.json'
try:
	s3.download_file(S3_BUCKET_INPUT, config_file_key,'./config.json')     
except botocore.exceptions.ClientError as e:
	if e.response['Error']['Code'] == "404":
		print(f'File:{config_file_key} Does not exist' )
		exit(-1)


# Read config file
print('Reading config.json')
with open("./config.json") as json_data_file:
	data = json.load(json_data_file)

userName = data["userName"]
projectName = data["projectName"]
projectType = data["projectType"]
print("User Name:",userName)
print("Project Name:", projectName)
print("Project Type:", projectType)


# Download files realated to username, project name and project type read from config.json
for key in s3.list_objects(Bucket = S3_BUCKET_INPUT,Prefix=f'{userName}/{projectName}/{projectType}')['Contents']:
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
			s3.download_file(S3_BUCKET_INPUT, key['Key'],'./user_data/'+key['Key'])


if projectType == 'image':
	train_image_model(userName=userName,projectName=projectName,projectType=projectType)
elif projectType == 'text':
	train_text_model(userName=userName,projectName=projectName,projectType=projectType)

# Delete config file so that training doesn't run on the same session
print(f'Deleting file {config_file_key} from s3')
s3.delete_object(Bucket=S3_BUCKET_INPUT,Key=config_file_key)
