# **Prem Kumar.K**  
*Data Scientist | Experience: 4+ yrs* 

### **Professional Experience:**
1. `Quantiphi Analytics Pvt Ltd` , *Mar,2021 - Present*
2. `Kryptos Technologies Pvt Ltd` , *Nov,2019 - Nov,2020*
3. `Tata Consultancy Services` , *Jan,2017 - Mar, 2019*

### **Education:**
* B.Tech in Mechanical Engineering, *Pondicherry University*, 2012-16, CGPA: *9.01/10*


### **Project 1: Chorus Detection**  
---
> **This project is a music retrieval system which is detecting the instances of chorus section in the music recording. Generally, chorus section in the audio represents the most repetitive part in the song.**

**`Tools:`** AWS EC2, Python, Docker, Flask.    
**`Technology:`**  Machine Learning, Audio Processing, Image Processing.  
**`Algorithm:`** DBSCAN.  

##### **Responsibility:**
* Given an audio, convert the audio into time-time similarity matrix and time-lag similarity matrix using  chorma features.
* Smoothing and binarize the similarity matrix using image processing techniques
* Finding the clusters using in the binarized image which represents the similar parts in the audio.
* Finding the matching chorus using python
* Trim the found choruses from the audio and split the audio into vocal component and Accompaniment component
* After splitting the audio, finding the presence of voice using Spectral Energy.
* The model is packaged into docker container and the model package is published in the AWS MarketPlace.


### **Project 2: Similar Image Retrieval** 
---
> **This project is a similar image retrieval system which is given an query image retrieving the similar images in the database of images.**

**`Tools:`**  AWS EC2, python, Elastic Search, AWS Sagemaker.   
**`Technology:`** Machine Learning, Deep Learning, Image Processing.    
**`Algorithm:`** K-Nearest Neighbor, ResNet.    

##### **Responsibility:**
* The task here is to given a query image retrieving all the similar images in the database.
* Used ResNet to find the encoding of all the available images in the dataset.
* Upload all the image feature encodings in the AWS ElasticSearch with unique id for each image.
* Now, given a query image, found the ResNet model to find out the encodings of the query image and used elastic search KNN to find the k nearest neighbors
* Those K neighbors will be the similar images for the given query image.

### **Project 3: Bags Classifier**  
---
> **This project focussed on image classification model which classifies the different type of bags of a single luxury brand inorder to automate the manual process.**

**`Tools:`**  AWS EC2, python, AWS Sagemaker, AWS Rekognition.   
**`Technology:`** Deep Learning, Image Processing.     
**`Algorithm:`** ResNet.    

##### **Responsibility:**
* The task here is to build a image classification model which classifies the given image
* Implemented different techniques to counter the imbalanced dataset as the data was imbalanced.
* Used the AWS Rekognition to train the model and got the  evaluation metrics and also train another AWS Sagemaker built-in image classification algorithm model and got the metrics for it too.
* Explained to the client about this two models, their corresponding metrics and the cost requirements of using this model and suggest which can be least expensive.
* Deployed the sagemaker model using sagemaker endpoint.

### **Project 4: MLOPS capabilities**
---
> **This work purely focussed on MLOPS platform of different cloud vendors and their respective CI/CD capabilites and implement the features needed in Quantiphi MLOPS platform.**

**`Tools:`** Python, AWS Sagemaker Pipelines, AWS Sagemaker Projects,  
             Vertex AI(GCP), Kubeflow, Airflow.    
**`Technology:`**  Machine Learning, Deep Learning.   

##### **Responsibility:**
* The task here is to research the MLOPS capabilities of AWS Sagemaker and GCP's Vertex AI.
* Implemented a basic model using Sagemaker pipelines and Sagemaker projects to understaned the inner workings of CI-CD on AWS platform
* Implemented a basic model using Vertex AI pipelines and Kubeflow pipeliens and the CI-CD capabilties in GCP.
* Implemented a processing step, Algorithm step to create a pipeline for computer vision model in the Quantiphi MLOPS platform and orchaestrated the workflow using Airflow.

### **Project 5: Pacing Rate Prediction**
---
> **This project focussed on predicting the pacing rate which helps to make the number of outbound calls in a call center in a bank in order to optimize the work force in the campaign.**

**`Tools:`** Python, MongoDB.    
**`Technology:`** Machine Learning, Time Series.     

##### **Responsibility:**
* The task here is to predict the pacing rate in order to automate the manual change of pacing rate in the call center application of the bank. 
* Pacing rate determines how many outbound calls to dial for the particular time.
* Used basic data explanatory analysis and time series analysis to find out the call acceptance pattern and time taken by the caller to accept and other features.
* Using those features, built a formula that can predict the pacing rate for the particular time.
* Built a script that can dynamically predict the pacing rate based on those features.
* Deployed the script in the production and used cron to scheduled based on needs.

### **Project 6: Emotion Detection model**
---
> **This project focussed on image classification model which classifies the different type of emotions of a human while playing the dart game in the entertainment center.**

**`Tools:`** AWS EC2, Python, Flask.   
**`Technology:`** Deep Learning, Image Processing.    
**`Algorithm:`** Xception Network.    

##### **Responsibility:**
* The task here is to predict the emotion of the players playing the dart game in entertainment center.
* Collected the images using the camera used to shoot the players while playing the game
* Used different augmentation techniques to create more data for the training
* Tried different face detection model to find out the best face detection system since the camera is nearly 2 metres away from the user.
* Trained an Xception model to classify human emotions in the particular frame of the video
* Used tflite to reduce the model size to deploy the model in the on-prem devices.
* Built a Flask API to predict emotion in the given image and sent the image for the next process of making the highlights video of the good moments during the play.

### **Project 7: Age and Gender Detection**
---
> **This project focusses on two image classification models where one model detects the gender and the other detects the age of the user for verification during the KYC on the bank.**

**`Tools:`** Python.   
**`Technology:`** Deep Learning, Image Processing.    
**`Algorithm:`** MobileNet for Face Detection, ResNet for Gender Detection.    

##### **Responsibility:**
* The task here is to predict the gender of the user and predicting the age of the user the verification during the KYC process.
* Collected dataset from various resources for both the application and received from client as well.
* Implemented different techniques to counter the imbalanced dataset as the data was imbalanced.
* Used multi-threading techniques to run both the model since one model doesn't depend on other.
* Deployed in the production on-prem devices in the bank.

### **Project 8: Stencil Number Extraction**
---
> **This project focussed on extracting the stencil numbers in the tires of an automobile.**

**`Tools:`** Python, AWS EC2.    
**`Technology:`** Machine Learning, Image Processing, Deep Learning.     
**`Algorithm:`** OCR.    

##### **Responsibility:**
* The task here is to extract the stencil number in the given image which contains tyres.
* Tried different image processing techniques like smoothing, binarizing, edge detection, skew correction, dewaring, etc to extract the words out of it.
* Tried to use AWS Textract and AWS rekognition to extract the words out of the image
* Tried OCR techniques like EasyOCR, pytesseract to extract the text out of the images.

### **Project 9: Extract and Verify Signatures**
---
> **This project focussed on extracting and verifying the signatures of a user in the bank.**

**`Tools:`** Python.   
**`Technology:`**  Machine Learning, Deep Learning, Image Processing.    
**`Algorithm:`** Yolov3, Siamese Network.   

##### **Responsibility:**
* The task here is to extract the signature and verify the signature with the original signature
* Trained Yolov3 to extract the signatures from the given image
* Trained a siamese network model to verify whether both the images are similar or not.

### **Project 10: Prediction of Best Tool**
---
> **This project focussed on predicting the best tool for the manufacturing process under given operating conditions.**

**`Tools:`** Python.   
**`Technology:`** Machine Learning, Natural Language Processing.    
**`Algorithm:`** Logistic Regression.    

##### **Responsibility:**
* The task here is to predict the best tool for the manufacturing process under certain operating conditions.
* Given the text document, built a features using regex and Natural language processing and save it as csv.
* Tried different machine learning algorithm to figure out which model performs better for our metric.
* Deployed the model into the production on on-prem devices.


 
