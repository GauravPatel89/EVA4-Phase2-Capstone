try:
    import unzip_requirements
except ImportError:
    pass

import json
import boto3
import os
import tarfile
import io
import base64
from requests_toolbelt.multipart import decoder

from torchtext import data
import torch
import spacy
import pickle

print('Import End....')
device = "cpu"

S3_MODEL_BUCKET = os.environ['S3_MODEL_BUCKET'] if 'S3_MODEL_BUCKET' in os.environ else 'gauravp-eva4-capstone-models'
s3 = boto3.client('s3')

nlp = spacy.load('/tmp/pkgs-from-layer/en_core_web_sm/en_core_web_sm-2.2.5')

def classify_text(input_text,model,tokenizer):

    # tokenize the text
    tokenized = [tok.text for tok in nlp.tokenizer(input_text)]
    # convert to integer sequence using predefined tokenizer dictionary
    indexed = [tokenizer[t] for t in tokenized]
    # compute no. of words
    length = [len(indexed)]
    # convert to tensor
    tensor = torch.LongTensor(indexed).to(device)
    # reshape in form of batch, no. of words
    tensor = tensor.unsqueeze(1).T
    # convert to tensor
    length_tensor = torch.LongTensor(length)
    # Get the model prediction
    prediction = model(tensor, length_tensor)

    _, pred = torch.max(prediction, 1)

    return pred.item()


def infer(event, context):
    try:

        print(event['body'])
        bodyTemp = event["body"]
        print("Body Loaded")
    
        body = json.loads(bodyTemp)
        print(body,type(body))
        input_text = body["text"]
        print(input_text)
        print(type(input_text))

        userName = body["userName"]
        print("Username:",userName)

        projectName = body["projectName"]
        print("Project Name:",projectName)


        # Check if model file and userProject files exist then only proceed with inference
        
        # Get all the files in 'gauravp-eva4-capstone-models' bucket
        print('Getting File list from ',S3_MODEL_BUCKET)
        files = [key['Key'] for key in s3.list_objects(Bucket = S3_MODEL_BUCKET)['Contents']]
        
        userFileName = f'{userName}_{projectName}_text.json'
        userModelFileName = f'{userName}_{projectName}_text.pt'
        tokenizerFileName = f'{userName}_{projectName}_text.pkl'
        
        userProjectExists = False
        numClasses = 0
        classNames = []
        prediction = '-1'
        predClassName = 'FAILED!!!'


        if ((userFileName in files) and (userModelFileName in files) and (tokenizerFileName in files)):
            userProjectExists = True

            # Download Tokanizer file
            tokenizerFilePath = '/tmp/' + tokenizerFileName

            if os.path.exists(tokenizerFilePath):
                os.remove(tokenizerFilePath) 

            print('Downloading TokanizerFile: ',tokenizerFileName)
            s3.download_file(S3_MODEL_BUCKET, tokenizerFileName,tokenizerFilePath)

            print('Loading Tokanizer')
            tokenizer_file = open(tokenizerFilePath, 'rb')
            tokenizer = pickle.load(tokenizer_file)

            print(f'{userFileName} file found in S3:{S3_MODEL_BUCKET}')
            userFilePath = '/tmp/' + userFileName

            if os.path.exists(userFilePath):
                os.remove(userFilePath) 

            print('Downloading user file: ',userFileName)
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
            model = torch.load(bytestream)
            print("Model Loaded...")

            prediction = classify_text(input_text=input_text,model=model,tokenizer=tokenizer)
            predClassName = classNames[prediction]
            print('Prediction: ',prediction)
            print('Predicted Class: ',predClassName)
            

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({'prediction': prediction,'PredictedClass': predClassName,'userProjectExists':userProjectExists })
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
