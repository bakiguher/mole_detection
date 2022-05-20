# Mole detection

- Repository: `mole_detection`
- Type of Challenge: `Consolidation`
- Duration: `8 days`
- Deadline: `24/05/2022 4:30` **(code)**
- Presentation: `25/05/2022 1:30 PM`
- Challenge: Solo


## Mission objectives

- Be able to apply a CNN in a real context
- Be able to preprocess data for computer vision
- Be able to evaluate your model (split dataset, confusion matrix, hyper-parameter tuning, etc)
- Be able to visualize your model results and evaluations (properly labeled, titled...)
- Be able to deploy your solution in an simple APP locally or on Heroku





<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#website">Website</a></li>
    <li><a href="#preprocess">Preprocess</a></li>
	 <li><a href="#models">Model</a></li>
    
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project


The purpose of the project is to develop a tool that would be able to detect moles that need to be handle by doctors.
The project will be available on a simple web page where the user could upload a picture of the mole and see the result.
The project will be upload on internet with flask, doker and heroku. 



### Built With

* [Python](https://www.python.org/)
* [Numpy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Scikit-learn](https://scikit-learn.org/)
* [TensorFlow](https://www.tensorflow.org/)
* [Docker](https://www.docker.com/)
* [Heroku](https://www.heroku.com/)



<!-- GETTING STARTED -->
## Getting Started

To work with this app, you have two options. Either work directly with the app at [this URL](https://moledetect.herokuapp.com/), either build it yourself from the sources and deploy it in a Docker container on Heroku as it is explained in the next subsection.

### Prerequisites

You'll need the packages/software described above.

### Installation

#### HEROKU

* **Install the Heroku CLI:**
  * The Heroku Command Line Interface (CLI) makes it easy to create and manage your Heroku apps directly from the terminal.
It’s an essential part of using Heroku.
  ```sh
  sudo snap install --classic heroku
  ```
* **Deployment on Heroku:**
  * Heroku favours Heroku CLI therefore using command line is (ensure the CLI is up-to-date) crucial at this step. 
  ```sh
  heroku login
  ```
  * After logging in to the respective Heroku account, the container needs to be registered with Heroku using 
  ```sh
  heroku container:login
  ```
  * Once the container has been registered, a Heroku repo would be required to push the container which could be created : 
  ```sh
  heroku create <yourapplicationname>
  ```
  **NOTE**: If there is no name stated after '_create_', a random name will be assigned.
  
  * When there is an application repo to push the container, it is time to push the container to web : 
  ```sh
  heroku container:push web --app <yourapplicationname>
  ```
  * Following the 'container:push' , the container should be released on web to be visible with 
  ```sh
  heroku container:release web --app <yourapplicationname>
  ```
  * If the container has been released properly, it is available to see using 
  ```sh
  heroku open --app <yourapplicationname>
  ```
  * Logging is also critical especially if the application is experiencing errors : 
  ```sh
  heroku logs --tail <yourapplicationname>
  ```


**IMPORTANT NOTE:** While with _localhost_ and _Docker_ it is not mandatory to specify the PORT, if one would like to deploy on Heroku, the port needs to be specified within the 'app.py' to avoid crashes.


## Preprocess

Data location : https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
Downloaded to data directory in project and extracted. Only HAM10000_metadata.csv and image files under HAM10000_images are used. In the extraction image files come in two folders they are combined in to one folder.  
More information about the dataset can be found at  https://challenge.isic-archive.com/


## Models
**2 models are created**
 - one for benign or malignant classification 
 - other one is for Type of the mole. 

**Mole types**
 - Melanocytic nevi (nv) : benign
 - Melanoma (mel): malignant
 - Benign keratosis-like lesions (bkl) : benign
 - Basal cell carcinoma (bcc): malignant
 - Actinic keratoses (akiec): benign
 - Vascular lesions (vasc): benign
 - Dermatofibroma (df): benign


## Repo Architecture 

```

│   README.md                     : This file
│   app.py                        : Flask app start
│   util.py                       : some utility functions  
│   Dockerfile                    : Docker file  
│   Procfile                      : Heroku start application- gunicorn
│   docker-compose.yml            : Docker file
|   requirements.txt              : Requirements file
|___             
│   data                          : data folder for image and csv file 
│   │                             : DOWNLOAD
│___   
|    model          
│   │ modelbm.h5                  : Weights of the model for benign/malignant classification.
│   │ modelmk.h5                  : Weights of the model for mole type classification.
│   │ modelbm.json                : Model structure for benign/malignant classification.
│   │ modelmk.json                : Model structure for mole type classification.

│___  
|    templates
│   | base.html                   : Template for flask  
│   | index.html                  : Template for flask  
│___  
|    static
│   | main.css                    : css file for flask  
│   | main.js                     : javascript file for flask  


```



Project Link: [https://github.com/bakiguher/mole_detection](https://github.com/bakiguher/mole_detection)


