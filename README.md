<!-- ![photo](mine.jpg)   -->
<img style="float: right;" src="imgonline-com-ua-resize-iMFL6lyMCZaC3__01.jpg">  

# **PREM KUMAR.K**
*Senior Data Scientist | Experience: 5+ yrs*  

#### **REACH ME:**
**`Email:`** premkumarkaliyamoorthy@gmail.com  
**`Contact:`** 8838787234  
**`LinkedIn:`** [https://www.linkedin.com/in/prem-kumar-870820133](https://www.linkedin.com/in/prem-kumar-870820133)

### **PROFESSIONAL EXPERIENCE:**
1. `Quantiphi Analytics Pvt Ltd` , *Mar,2021 - Present*
2. `Kryptos Technologies Pvt Ltd` , *Nov,2019 - Nov,2020*
3. `Tata Consultancy Services` , *Jan,2017 - Mar, 2019*

### **EDUCATION:**
* B.Tech in Mechanical Engineering, *Pondicherry University*, 2012-16, CGPA: *9.01/10*

### **SKILLS:**
1. `Languages       :` Python.
2. `Frameworks      :` Flask, Scikit-Learn, Keras, Tensorflow, river.
3. `Tools           :` Git, SQL, Sagemaker, Docker, Airflow, MongoDB, DVC, Mlflow, ElasticSearch.
4. `Cloud           :` AWS & its services.
5. `Technologies    :` Machine Learning, Deep Learning, Computer Vision, NLP.
6. `Major Libraries :` Numpy, Pandas, Matplotlib, Seaborn, cv2, NLTK, Librosa, BeautifulSoup.

### **PROJECT 1: CHORUS DETECTION**  
---
> **This project is a music retrieval system that detects instances of chorus sections in the music recording. Generally, the chorus section in the audio represents the most repetitive part of the song.**

![chorus detection](kelly-sikkema-X-etICbUKec-unsplash.jpg)  
Photo by <a href="https://unsplash.com/@kellysikkema?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Kelly Sikkema</a> on <a href="https://unsplash.com/s/photos/audio?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

**`Tools      :`** AWS EC2, Python, Docker, Flask.    
**`Technology :`** Machine Learning, Audio Processing, Image Processing.  
**`Algorithm  :`** DBSCAN.  

##### **RESPONSIBILITIES:**
* Given audio, convert the audio into a time-time similarity matrix and a time-lag similarity matrix using chorma features.
* Smoothing and binarizing the similarity matrix using image processing techniques.
* Finding the clusters in the processed image that represent the similar parts in the audio.
* Find the matching chorus, trim the found choruses from the audio, and split the audio into a vocal component and an accompaniment component.
* Finding the presence of a voice using spectral energy
* I packaged the model into a Docker container, and it is published in the AWS MarketPlace.


### **PROJECT 2: SIMILAR IMAGES RETRIEVAL** 
---
> **This project is a similar image retrieval system that, given a query image, retrieves similar images from the database of images.**

![image_similarity](dietmar-ludmann-qs4j-39TaBQ-unsplash.jpg)  
Photo by <a href="https://unsplash.com/@d13n?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Dietmar Ludmann</a> on <a href="https://unsplash.com/s/photos/cats?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

**`Tools      :`** AWS EC2, python, Elastic Search, AWS Sagemaker.   
**`Technology :`** Machine Learning, Deep Learning, Image Processing.    
**`Algorithm  :`** K-Nearest Neighbor, ResNet.    

##### **RESPONSIBILITIES:**
* The task here is, given a query image, to retrieve all the similar images in the database.
* We used ResNet to find the encoding of all the available images in the dataset.
* Uploaded all the image feature encodings in the AWS ElasticSearch with unique ids for each image.
* Given a query image, find the encoding using the ResNet model and use elastic search KNN to find the k nearest neighbours.

### **PROJECT 3: BAGS CLASSIFIER**  
---
> **Focused on an image classification model that classifies the different types of bags from a single luxury brand in order to automate the manual process.**

![bags_classifier](diana-akhmetianova-nvQemFKRBUo-unsplash.jpg)  
Photo by <a href="https://unsplash.com/@dreamcatchlight?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Diana Akhmetianova</a> on <a href="https://unsplash.com/s/photos/bags?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

**`Tools      :`** AWS EC2, python, AWS Sagemaker, AWS Rekognition.   
**`Technology :`** Deep Learning, Image Processing.     
**`Algorithm  :`** ResNet.    

##### **RESPONSIBILITIES:**
* The task here is to build an image classification model that classifies the given image.
* Implemented different techniques to counter the imbalanced dataset as the data was imbalanced.
* Have used AWS Rekognition and the AWS Sagemaker built-in image classification algorithm model to train the model and got the evaluation metrics for both of the models.
* Explain to the client about these two models, their corresponding metrics, and the cost requirements of using them, and suggest which is the least expensive.
* Deployed the Sagemaker model using the Sagemaker endpoint

### **PROJECT 4: MLOPS PLATFORM**
---
> **Worked in an agile environment to build a MLOps platform where we built and tested an entire ML lifecycle using Sagemaker BYOC, pipelines using Airflow for different ML use cases, and helped organizations migrate their use cases to our platform.**

![MLOPS](glenn-carstens-peters-npxXWgQ33ZQ-unsplash.jpg)  
Photo by <a href="https://unsplash.com/@glenncarstenspeters?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Glenn Carstens-Peters</a> on <a href="https://unsplash.com/s/photos/machine-learning?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

**`Tools      :`** Python, Sagemaker Pipelines, Sagemaker Projects, Vertex AI(GCP), Kubeflow, Airflow, Docker.   
**`Technology :`** Machine Learning, Deep Learning.   

##### **Responsibility:**
* Created a components for each ML lifecycle like processing, training, explainability, and monitoring using Sagemaker BYOC and Docker.
* Created a different pipelines for different tasks like training, retraining, inference, creating endpoints, and monitoring using Airflow.
* Built a sample projects with each use case, like tabular, computer vision, and NLP to test the entire ML lifecycle using the platform we have created.
* Documented the user manual for do's and dont's of components, pipelines, and their inner workings and how to create them on their own for their own use cases.
* Helped organizations migrate their projects into our platform and gave them deep dive sessions on how to migrate them on their own.

### **PROJECT 5: PACING RATE PREDICTION**
---
> **This project focused on predicting the pacing rate, which helps to determine the number of outbound calls in a call center in a bank, in order to optimize the work force in the campaign.**

![pacing_rate](austin-distel-97HfVpyNR1M-unsplash.jpg)  
Photo by <a href="https://unsplash.com/@austindistel?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Austin Distel</a> on <a href="https://unsplash.com/s/photos/call-center?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

**`Tools      :`** Python, MongoDB.    
**`Technology :`** Machine Learning, Time Series.     

##### **RESPONSIBILITIES:**
* The task here is to predict the pacing rate in order to automate the manual change of the pacing rate in the call center application of the bank. 
* The pacing rate determines how many outbound calls to dial at a particular time.
* Did explanatory data analysis and time series analysis to find out the call acceptance pattern, time taken by the caller to accept, and other features.
* Built a formula that can predict the pacing rate and created a script that can dynamically predict the pacing rate for a particular time using those features.
* Deployed the script in production and used cron to schedule it.

### **PROJECT 6: EMOTION DETECTION MODEL**
---
> **This project focused on an image classification model that classifies the different types of emotions of a human while playing the dart game in the entertainment center.**

![emotion_detection](tengyart-dTgyj9okQ_w-unsplash.jpg)  
Photo by <a href="https://unsplash.com/@tengyart?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Tengyart</a> on <a href="https://unsplash.com/s/photos/emotion?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

**`Tools      :`** AWS EC2, Python, Flask.   
**`Technology :`** Deep Learning, Image Processing.    
**`Algorithm  :`** Xception Network.    

##### **RESPONSIBILITIES:**
* The task here is to predict the emotions of the players playing the dart game in the entertainment center.
* Collected the images using the camera used to shoot the players while playing the game.
* Used different augmentation techniques to create more data for the training.
* Tried different face detection models to find out the best face detection system since the camera is nearly 2 meters away from the user.
* Trained an Xception model to classify human emotions in the particular frame of the video.
* Used tflite to reduce the model size to deploy the model on the on-prem devices.
* Built a Flask API to predict emotion in the given image and sent the image for the next process of making the highlights video of the good moments during the play.

### **PROJECT 7: AGE & GENDER DETECTION**
---
> **This project focuses on two image classification models, where one model detects the gender and the other detects the age of the user for verification during the KYC process in the bank.**  

![age_gender_det](nathan-dumlao-4_mJ1TbMK8A-unsplash.jpg)  
Photo by <a href="https://unsplash.com/@nate_dumlao?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Nathan Dumlao</a> on <a href="https://unsplash.com/s/photos/family?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

**`Tools      :`** Python.   
**`Technology :`** Deep Learning, Image Processing.    
**`Algorithm  :`** MobileNet for Face Detection, ResNet for Gender Detection.    

##### **RESPONSIBILITIES:**
* The task here is to predict the gender and age of the user to verify during the KYC process.
* Collected dataset from various resources for both the application and the client as well.
* Implemented different techniques to counter the imbalanced dataset as the data was imbalanced.
* Used multi-threading techniques to run both the model and deployed the model in the production on-prem devices in the bank.

### **PROJECT 8: STENCIL NUMBER EXTRACTION**
---
> **This project focused on extracting the stencil numbers from the tires of an automobile.**

![stencil_number](nazim-zafri-4dNGik9Itfg-unsplash.jpg)  
Photo by <a href="https://unsplash.com/@nazimzafri?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Nazim Zafri</a> on <a href="https://unsplash.com/s/photos/tyre?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

**`Tools      :`** Python, AWS EC2.    
**`Technology :`** Machine Learning, Image Processing, Deep Learning.     
**`Algorithm  :`** OCR.    

##### **RESPONSIBILITIES:**
* The task here is to extract the stencil number from the given image, which contains tires.
* Tried different image processing techniques like smoothing, binarizing, edge detection, skew correction, dewaring, etc. to extract the words out of it.
* Tried to use AWS Textract and AWS rekognition to extract the words out of the image.
* Tried OCR techniques like EasyOCR and Pytesseract to extract the text out of the images.

### **PROJECT 9: EXTRACT & VERIFY SIGNATURES**
---
> **This project focused on extracting and verifying the signatures of a user in the bank.**

![signature_verify](signature-pro-j5Qbe2TG6MU-unsplash.jpg)  
Photo by <a href="https://unsplash.com/@signaturepro?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Signature Pro</a> on <a href="https://unsplash.com/s/photos/signature?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

**`Tools      :`** Python.   
**`Technology :`** Machine Learning, Deep Learning, Image Processing.    
**`Algorithm  :`** Yolov3, Siamese Network.   

##### **RESPONSIBILITIES:**
* The task here is to extract the signature and verify the signature with the original signature.
* Trained Yolov3 to extract the signatures from the given image
* Trained a siamese network model to verify whether both images are similar or not.

### **PROJECT 10: PREDICTION OF BEST TOOL**
---
> **This project focused on predicting the best tool for the manufacturing process under given operating conditions.**

![Best_tool](greg-rosenke-xoxnfVIE7Qw-unsplash.jpg)  
Photo by <a href="https://unsplash.com/@greg_rosenke?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Greg Rosenke</a> on <a href="https://unsplash.com/s/photos/manufacturing?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

**`Tools      :`** Python.   
**`Technology :`** Machine Learning, Natural Language Processing.    
**`Algorithm  :`** Logistic Regression.    

##### **RESPONSIBILITIES:**
* The task here is to predict the best tool for the manufacturing process under certain operating conditions.
* Given the text document, extract a feature using regex and natural language processing and save it as a CSV.
* Tried different machine learning algorithms to figure out which model performed better for our metric.
* Deployed the model into production on on-prem devices.

### **PROJECT 11: Forecasting of Resource Utilization**
---
> **This project focused on forecasting resource utilization for the telecom industry.**

**`Tools      :`** Python, AWS Forecast.   
**`Technology :`** Machine Learning, Time Series Forecasting.    

##### **RESPONSIBILITIES:**
* The task here is to forecast the resource utilization for the given time interval.
* Given the csv files that contain data on the utilization of the past. 
* Tried different preprocessing techniques using Pandas and saved it the way AWS Forecasting demands it.
* Build different models using AWS Forecasting to build forecasting models and compare the results.

### **PROJECT 12: Migration of a machine learning model from GCP to AWS environment**
---
> **This project focused on the migration of ML models from the GCP environment to the AWS environment.**

**`Tools      :`** Python, AWS Sagemaker, Docker.   
**`Technology :`** Machine Learning.    

##### **RESPONSIBILITIES:**
* The task here is to migrate the machine learning model that was deployed in the GCP environment to the AWS environment.
* Used Sagemaker BYOC concept to migrate the model from GCP to AWS. 
* Sagemaker BYOC expects a Docker container that contains the environment for the model to be deployed, so we created a Dockerfile that does that.
* Once the Docker image has been created, the Sagemaker endpoint has been created using the Docker image and the model from the GCP environment.
