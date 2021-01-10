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

### How to use above website?
There are two operation modes

#### 1. Image Classification
### CodeBase

#### Files:
1. 'endGameEnv.py' :  
This file contains Environment Class definition for simulation of the Car Navigation Game. For ease of use the environment class has been defined on the lines of standard Gym environments i.e. class has methods like reset(),step(),render(),sampleActionSpace(). 

2. 'endGameModels.py' :  
This file contains definition of Actor and Critic DNNs, ReplayBuffer used for storing step transitions and TD3 class which implements TD3 algorithm.

3. 'endGameInference.py' :  
This file contains code for evaluating the trained models. This is done by instantiating TD3 class object, loading saved Actor and Critic models and repeatedly generating action and taking action on defined environment to generate visualization.

4. 'endGameUtilities.py' :  
This file contains some of the utilities used by other files.

5. 'endGameEnvAllGoals.py' :  
This file has Environment definition similar to 'endGameEnv.py'. Difference is 'endGameEnv.py' environment's episode runs for only 3 random goal values while environment defined in this file runs the episode untill all the goals in the goalList have been achieved or car gets stuck to boundaries. This file is useful for evaluating how model is working for all the goal values. It was used for generating the submission video.

6. 'endgameTD3.py' :  
This file contains code for TD3 training. It combines components in other files to create the main training loop.

7. 'endGameTD3.ipynb'
This is the Google Colab file for TD3 training. 'endgameTD3.py' is simple .py version of this file. This file can be accessed on Google Colab [here.](https://colab.research.google.com/drive/1S3kT0hJlK4Uzh10DrAFE55l9OtZAbLyc?usp=sharing)

#### Directories
1. 'pytorch_models':  
This directory contains best saved models for Actor and Critic. Example code for evaluating and generating a video for this models can be found in ['endGameInference.py'](https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/EndGame/endGameInference.py). 

2. 'images':  
This directory contains image files used by the carEndgameEnv environment. It has following files
    
    a. 'car.png': Used for visualization of Car on city map. 
    b. 'citymap.png' : Used as city map
    c. 'MASK1.png': Used for representing road and non-road pixels in city map

3. 'Figures':  
This directory contains support files for README.md .

4. 'pytorch_models':
This directory contains best trained Actor, Critic models used for generating submission video.

    
### References  
1. [Fujimoto, S., van Hoof, H., and Meger, D. Addressing function approximation error in actor-critic methods. arXiv preprint arXiv:1802.09477, 2018.](https://arxiv.org/pdf/1802.09477.pdf)
2. [OpenAI — Spinning Up](https://spinningup.openai.com/en/latest/algorithms/td3.html) 
3. [Solving-CarRacing-with-DDPG](https://github.com/lzhan144/Solving-CarRacing-with-DDPG/blob/master/TD3.py)
4. [TD3: Learning To Run With AI](https://towardsdatascience.com/td3-learning-to-run-with-ai-40dfc512f93)
    

