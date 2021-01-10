# EVA4-Phase2-Capstone

## Problem Statement:  
Develop a [lobe.ai](https://lobe.ai/) clone:
- allow people to upload their own data:
     - Images (10x2 - 80:20) upto (100*10 - 70-30)
     - text samples (csv files) (min 100 - max 1000)*2-3 - 70:30)
- plug-in a model of your choice
- train the model on AWS EC2, use lambda to preprocess, trigger EC2 and match user id with the files stored on EC2
- move the model to lambda and show inferencing
### [Submission:](http://gauravp-eva4-capstone-website.s3-website.ap-south-1.amazonaws.com/)
Click below image to head over to Final Submission Website.
[![Capstone Final Submission](https://github.com/GauravPatel89/EVA4-Phase2-Capstone/blob/main/Figures/project_website_front.png)](http://gauravp-eva4-capstone-website.s3-website.ap-south-1.amazonaws.com/)

#### Group Members
Gaurav Patel (gaurav4664@gmail.com)

Chirag Saraiya (chirag2saraiya@gmail.com)

### How to use above website?
There are two operation modes

#### 1. Image Classification  
In this mode User can train a model for Image classification.

There are 3 options provided to users. 

- Train 
- Fetch Details 
- Inference

##### a. Train
i. Input User name, project name and  Number of classes. As soon as user inputs number of classes, as many number of class name fields and file upload buttons will show up.

<p align="center">
 <img width="800" height="400" src="https://github.com/GauravPatel89/EVA4-Phase2-Capstone/blob/main/Figures/image_training_1.png">
</p>
     
ii. Enter class names for each of the class fields and select training files for each class files. Press "Train" button. This will trigger a lambda and pass uploaded files to it. Lambda will in turn upload the files in proper directory structure to AWS S3 and start EC2 instance. 

<p align="center">
 <img width="800" height="400" src="https://github.com/GauravPatel89/EVA4-Phase2-Capstone/blob/main/Figures/image_training_2.png">
</p>

##### b. Fetch Details
i. Input User name, project name

<p align="center">
 <img width="800" height="400" src="https://github.com/GauravPatel89/EVA4-Phase2-Capstone/blob/main/Figures/image_fetch_user_1.png">
</p>
     
ii. Press "Fetch User Info" button. This will trigger a lambda and pass Username and Projectname. Lambda will check whether model corresponding to Username and Projectname exists and reverts back. The website will fill up fields like "Number of Classes" and all the class names.

<p align="center">
 <img width="800" height="400" src="https://github.com/GauravPatel89/EVA4-Phase2-Capstone/blob/main/Figures/image_fetch_user_2.png">
</p>

##### c. Inference
i. Upload Image file for classification and press "Infer" button. This will trigger a Lambda pass uploaded file as well as Username and Projectname to it. Lambda will fetch the relevant model from S3 and perform inference for uploaded file.    

<p align="center">
 <img width="800" height="400" src="https://github.com/GauravPatel89/EVA4-Phase2-Capstone/blob/main/Figures/image_inference_1.png">
</p>
     


#### 1. Text Classification  
In this mode User can train a model for Text classification.

There are 3 options provided to users. 

- Train 
- Fetch Details 
- Inference

##### a. Train
i. Input User name, project name and  Number of classes(max 3 classes). As soon as user inputs number of classes, as many number of class name fields and file upload buttons will show up.

<p align="center">
 <img width="800" height="400" src="https://github.com/GauravPatel89/EVA4-Phase2-Capstone/blob/main/Figures/text_training_1.png">
</p>
     
ii. Enter class names for each of the class fields and select training files for each class. Press "Train" button. This will trigger a lambda and pass uploaded files to it. Lambda will in turn upload the files in proper directory structure to AWS S3 and start EC2 instance. 

<p align="center">
 <img width="800" height="400" src="https://github.com/GauravPatel89/EVA4-Phase2-Capstone/blob/main/Figures/text_training_2.png">
</p>

##### b. Fetch Details
i. Input User name, project name

<p align="center">
 <img width="800" height="400" src="https://github.com/GauravPatel89/EVA4-Phase2-Capstone/blob/main/Figures/text_fetch_user_1.png">
</p>
     
ii. Press "Fetch User Info" button. This will trigger a lambda and pass Username and Projectname. Lambda will check whether model corresponding to Username and Projectname exists and reverts back. The website will fill up fields like "Number of Classes" and all the class names.

<p align="center">
 <img width="800" height="400" src="https://github.com/GauravPatel89/EVA4-Phase2-Capstone/blob/main/Figures/text_fetch_user_2.png">
</p>

##### c. Inference
i. Input the text for classification and press "Infer" button. This will trigger a Lambda pass text, Username and Projectname to it. Lambda will fetch the relevant model from S3 and perform inference for Input text.    

<p align="center">
 <img width="800" height="400" src="https://github.com/GauravPatel89/EVA4-Phase2-Capstone/blob/main/Figures/text_inference_1.png">
</p>

### CodeBase

#### Directories
1. 'deployment':  
This directory contains deployment files for various lambda functions used in the project. We have used 3 Lambda functions.
- ["gp-eva-capstone-project"](https://github.com/GauravPatel89/EVA4-Phase2-Capstone/tree/main/deployment/gp-eva-capstone-project)
This lambda is triggered by the website when user wants to train her model. It gathers all the images passed by the website and uploads them to AWS S3 in a neat order according to the Username,Projectname and Class names. It also generates a 'config.json' file to convey training information to EC2. Finally it start the EC2 Instance.

- ["gp-eva-capstone-project-inference"](https://github.com/GauravPatel89/EVA4-Phase2-Capstone/tree/main/deployment/gp-eva-capstone-project-inference)
This lamda does the inference work for Image Classification mode. It is triggered by the website when user wants to perform inference on some image. Based on the Username and Projectname it fetches the relevant model from S3 and performs inference for user uploaded file on the loaded model.

- ["gp-eva-capstone-text-inference"](https://github.com/GauravPatel89/EVA4-Phase2-Capstone/tree/main/deployment/gp-eva-capstone-text-inference)
This lamda does the inference work for Text Classification mode. It is triggered by the website when user wants to perform inference on some text. Based on the Username and Projectname it fetches the relevant model from S3 and performs inference for user input text on the loaded model.

2. 'images':  
