# Mole detection

  

- Repository: `mole_detection`

- Type of Challenge: `Consolidation`

- Duration: `8 days`

- Deadline: `24/05/2022 4:30`  **(code)**

- Presentation: `25/05/2022 1:30 PM`

- Challenge: Solo

  
  

## Mission objectives

  

- Be able to apply a CNN in a real context

- Be able to preprocess data for computer vision

- Be able to evaluate your model (split dataset, confusion matrix, hyper-parameter tuning, etc)

- Be able to visualize your model results and evaluations (properly labeled, titled...)

- Be able to deploy your solution in an simple APP locally or on Heroku

  
  
  

<!-- TABLE OF CONTENTS -->

<details  open="open">

<summary>Table of Contents</summary>

<ol>

<li><a  href="#about-the-project">About The Project</a></li>

<li><a  href="#getting-started">Getting Started</a></li>

<li><a  href="#preprocess">Preprocess</a></li>

<li><a  href="#model">Model</a></li>

</ol>

</details>

  
  
  

<!-- ABOUT THE PROJECT -->

## About The Project

  
  

The purpose of the project is to develop a tool that would be able to detect moles that need to be handle by doctors.

The project will be available on a web app where the user could upload a picture of the mole and see the result.

The project will be upload on internet with flask, doker and heroku.
 

#### Built With

  

* [Python](https://www.python.org/)
* [Numpy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Scikit-learn](https://scikit-learn.org/)
* [TensorFlow](https://www.tensorflow.org/)
* [Docker](https://www.docker.com/)

 
  
  

<!-- GETTING STARTED -->

## Getting Started

  

To work with this app, you have two options. Either work directly with the app at [this URL](https://moledetect.herokuapp.com/), or build it yourself from the sources and deploy it in a Docker container on Heroku as it is explained in the next subsection.

  

### Prerequisites
 

You'll need the packages/software described above.

 

### Installation

    Create the model you choose with jupyter file.
    docker-compose build
    docker-compose up


## Preprocess

  

Data location : https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

Download the zip file and extract it to data directory in the project folder. Only HAM10000_metadata.csv and image files under HAM10000_images are used. In the extraction image files come in two folders they are combined in to one folder.

More information about the dataset can be found at https://challenge.isic-archive.com/

**Mole types**

- Melanocytic nevi (nv) : benign

- Melanoma (mel): malignant

- Benign keratosis-like lesions (bkl) : benign

- Basal cell carcinoma (bcc): malignant

- Actinic keratoses (akiec): benign

- Vascular lesions (vasc): benign

- Dermatofibroma (df): benign
  

## Model

**2 models are created**

- one for benign or malignant classification

- other one is for Type of the mole. This one is used on app

**Data is imbalanced I used different aproaches** 
  *1- class weights library from sklearn*:  Training the model took just a few minutes but classification report on test data was not good.
      class weights:{0: 4.37, 1: 2.78, 2: 1.30, 3: 12.44, 4: 1.28, 5: 0.21, 6: 10.07}

![classification report](/visuals/cr_weights.jpg)


  *2-Randomoversampling from imblearn library*: Training takes around 30 minutes, classification report looks quite good but I believe model saw all the data avaliable and it memorized everything. It has 0.94 precision on test data.


Classification Report

![classification report](/visuals/cr_oversampling.jpg)

Confusion Matrix :

![confusion matrix](/visuals/cm_oversampling.jpg)

Incorrect Classification Percentages:

![incorrect](/visuals/incr_rand_overs.jpg)

  *3-Resampling from sklearn library* Training takes around 5 minutes Accuracy is 0.90 but presicion is at 0.75 

![classification report](/visuals/cr_nsample.jpg)
  

I cleaned the hair on the moles with opencv. Loading 10000 images to memory may cause some problems 
	


## Repo Architecture

  

```

  

│ README.md           : This file
│ app.py              : Flask app start
│ util.py             : some utility functions
│ Dockerfile          : Docker file
│ Procfile            : Heroku start application- gunicorn
│ docker-compose.yml  : Docker file
| requirements.txt    : Requirements file
| modelbm.ipynb       : Model creation for benign or malignant
| modeltype.ipynb     : Model creation for mole type
|___
│ data                : data folder for image and csv file
│ │ : DOWNLOADAD FROM KAGGLE
│___
| model
│ │ modelbm.h5        : Weights of the model for benign/malignant classification.
│ │ modelmk.h5        : Weights of the model for mole type classification.
│ │ modelbm.json      : Model structure for benign/malignant classification.
│ │ modelmk.json      : Model structure for mole type classification.
│___
| templates
│ | base.html         : Template for flask
│ | index.html        : Template for flask
│___
| static
│ | main.css          : css file for flask
│ | main.js           : javascript file for flask
│___
| visuals
| | : IMAGES OF SOME REPORTS
  
  

```

  
  
  

Project Link: [https://github.com/bakiguher/mole_detection](https://github.com/bakiguher/mole_detection)