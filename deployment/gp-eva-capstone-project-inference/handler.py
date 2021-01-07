try:
    import unzip_requirements
except ImportError:
    pass

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

import boto3
import os
import tarfile
import io
import base64
import json

from requests_toolbelt.multipart import decoder

print('Import End....')


S3_MODEL_BUCKET = os.environ['S3_MODEL_BUCKET'] if 'S3_MODEL_BUCKET' in os.environ else 'gauravp-eva4-capstone-models'
s3 = boto3.client('s3')

def transform_image(image_bytes):
    try:
        print('transform_image: start')
        transformations = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = Image.open(io.BytesIO(image_bytes))
        print('transform_image: Image opened')
        return transformations(image).unsqueeze(0)
    except Exception as e:
        print('transform_image: start',repr(e))
        raise(e)


def get_prediction(model,image_bytes):
    print('get_prediction: start')
    tensor = transform_image(image_bytes=image_bytes)
    print('Prediction: ',tensor)
    return model(tensor).argmax().item()


def infer(event, context):
    try:

        print('start')
        content_type_header = event['headers']['content-type']
        #print('content_type_header',content_type_header)
        body = base64.b64decode(event["body"])
        print('Body loaded')

        
        userName = decoder.MultipartDecoder(body, content_type_header).parts[0].content.decode("utf-8")
        print("Username:",userName)

        projectName = decoder.MultipartDecoder(body, content_type_header).parts[1].content.decode("utf-8")
        print("Project Name:",projectName)

        picture = decoder.MultipartDecoder(body, content_type_header).parts[2].content
        print("Picture obtained.")

        # Check if model file and userProject files exist then only proceed with inference
        
        # Get all the files in 'gauravp-eva4-capstone-models' bucket
        print('Getting File list from ',S3_MODEL_BUCKET)
        files = [key['Key'] for key in s3.list_objects(Bucket = S3_MODEL_BUCKET)['Contents']]
        
        userFileName = f'{userName}_{projectName}.json'
        userModelFileName = f'{userName}_{projectName}.pt'
        userProjectExists = False
        numClasses = 0
        classNames = []
        prediction = '-1'
        predClassName = 'FAILED!!!'


        if ((userFileName in files) and (userModelFileName in files)):
            userProjectExists = True
            print(f'{userFileName} file found in S3:{S3_MODEL_BUCKET}')
            userFilePath = '/tmp/' + userFileName

            if os.path.exists(userFilePath):
                os.remove(userFilePath) 

            s3.download_file(S3_MODEL_BUCKET, userFileName,userFilePath)
            with open(userFilePath, "r") as inFile:
                userProjectInfo = json.load(inFile)

            print('User Info:',userProjectInfo)
                
            numClasses = userProjectInfo['numClasses']
            classNames = userProjectInfo['classNames']

            print('Getting Model: ',userModelFileName)

            obj = s3.get_object(Bucket=S3_MODEL_BUCKET, Key=userModelFileName)
            print("get model: Creating Bytestream...")
            bytestream = io.BytesIO(obj['Body'].read())
            print("get model: Loading Model...")
            model = torch.jit.load(bytestream)
            print("Model Loaded...")

            prediction = get_prediction(model=model,image_bytes=picture)
            print('Id',prediction)
            predClassName = classNames[prediction]
            print('Id',prediction,' Class:',predClassName)

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({'predictedId': prediction,'predictedClass':predClassName,'numClasses':numClasses,'userProjectExists':userProjectExists })
        }
    except Exception as e:
        print('classify_image',repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }
