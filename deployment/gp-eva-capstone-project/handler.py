#try:
#    import unzip_requirements
#except ImportError:
#    pass

#import torch
#import torchvision
#import torchvision.transforms as transforms
#from PIL import Image

import boto3
import os
#import tarfile
import io
import base64
import json
import sys

print("Present Working Directory",os.system('pwd'))
print("List of files in Directory",os.system('ls'))

sys.path.insert(1, '/var/task/package')
from requests_toolbelt.multipart import decoder

region = 'ap-south-1'
instances = ['i-09ea0032860e37acb']
ec2 = boto3.client('ec2', region_name=region)


S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'gauravp-eva4-capstone-bucket'
S3_MODEL_BUCKET = 'gauravp-eva4-capstone-models'
s3 = boto3.client('s3')

def hello(event, context):
    try:

        

        print('start')
        content_type_header = event['headers']['content-type']
        print('content_type_header',content_type_header)
        body = base64.b64decode(event["body"])
        print('Body loaded')

        
        userName = decoder.MultipartDecoder(body, content_type_header).parts[0].content.decode("utf-8")
        print("Username:",userName)

        projectName = decoder.MultipartDecoder(body, content_type_header).parts[1].content.decode("utf-8")
        print("Project Name:",projectName)

        currentMode = decoder.MultipartDecoder(body, content_type_header).parts[2].content.decode("utf-8")
        print("Current Mode:",currentMode)

        if currentMode == 'userInfo':
            # Get all the files in 'gauravp-eva4-capstone-models' bucket
            files = [key['Key'] for key in s3.list_objects(Bucket = S3_MODEL_BUCKET)['Contents']]
            userFileName = f'{userName}_{projectName}.json'
            userProjectExists = False
            numClasses = 0
            classNames = []
            if userFileName in files:
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

            
            return {
                "statusCode": 200,
                "headers": {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    "Access-Control-Allow-Credentials": True
                },
                'body': json.dumps({'userProjectExists': userProjectExists , 'numClasses': numClasses, 'classNames':classNames})
            }


        numClasses = int(decoder.MultipartDecoder(body, content_type_header).parts[3].content.decode("utf-8"))
        print("Number Of Classes", numClasses)

        print("Fetching Deccoder Object")
        decoderObj = decoder.MultipartDecoder(body, content_type_header)

        count = 0
        print("Files uploaded ")
        for part in decoderObj.parts:  
            print(part.headers[b'Content-Disposition'].decode())

            if count < (4):  # first (3) objects are config parameters so ignore
                print(count)
                
                count += 1
                continue
            
            fileName = part.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1].strip('\"')
            className = part.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1].strip('\"')
            print(f'File #{count}: {fileName} from {className}')
            count += 1
            s3.put_object(Bucket=S3_BUCKET, Key=f'{userName}/{projectName}/train_data/{className}/{fileName}', Body=part.content)


        print(f'Number of Fields received = {count}')

        print("Creating config file")
        config_dict = { "userName":userName,"projectName":projectName}

        with open("/tmp/config.json", "w") as outfile:
            json.dump(config_dict, outfile)

        print("Uploading config file to S3")
        s3.upload_file('/tmp/config.json', S3_BUCKET, 'config.json')
        
    
        print('Starting EC2 instance')
        ec2.start_instances(InstanceIds=instances)
        print('started your instances: ' + str(instances))

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            'body': json.dumps({'UserName':userName,'projectName':projectName, 'numClasses': numClasses,'numParts':count})
        }
    except Exception as e:
        print('hello',repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            'body': json.dumps({"error": repr(e)})
        }
