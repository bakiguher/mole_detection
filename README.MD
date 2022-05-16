# Mole detection

- Repository: `mole_detection`
- Type of Challenge: `Consolidation`
- Duration: `8 days`
- Deadline: `24/05/2022 4:30` **(code)**
- Presentation: `25/05/2022 1:30 PM`
- Challenge: Solo

![AI care!](./assets/ai-care.jpg)

## Mission objectives

- Be able to apply a CNN in a real context
- Be able to preprocess data for computer vision
- Be able to evaluate your model (split dataset, confusion matrix, hyper-parameter tuning, etc)
- Be able to visualize your model results and evaluations (properly labeled, titled...)
- Be able to deploy your solution in an simple APP locally or on Heroku

## The Mission

The health care company "skinCare" hires you as a freelance data scientist for a short mission.
Seeing that they pay you an incredible amount of money you accept. Here is the mail you receive:

```text
from: sales.skincare.be@gmail.com
to:   projects@becode.org

Hi,

At skinCare we have more and more demands from companies for a tool that would be able to detect moles that need to be handled by doctors.

Your mission, if you accept it, would be to create an AI that can detect when the mole is dangerous. To be able to use your AI, we want you to create a simple web page where the user could upload a picture of the mole and see the result.

We don't have a web development department in our company, so you will need to handle the UI as well, we don't care about having something amazing on the design side, we just want a proof of concept to show our client that the product has potential. You are free in the technology you want to use.

You will also need to put your application on the internet so we can use it. I guess you know what that means.

You will find attached to this email the data you need.

If you have any questions, feel free to reply to this mail or send a mail to our department at the following email address: sales.skincare.be@gmail.com

Good luck,
skinCare sales team
```

Analyze the customer's request.
The dataset provided by the client can be found here: https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000

Possible disease states are "Melanoma", "Melanocytic nevus", "Basal cell carcinoma", "Actinic keratosis / Bowen’s disease (intraepithelial carcinoma)", "Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)", "Dermatofibroma", and "Vascular lesion". Approximately 10,000 images provided for training, 200 for validation, 1500 for test.



Melanocytic nevi (nv)
Melanoma (mel)
Benign keratosis-like lesions (bkl)
Basal cell carcinoma (bcc)
Actinic keratoses (akiec)
Vascular lesions (vasc)
Dermatofibroma (df)

value counts:
0    6705
1    1113
2    1099
3     514
4     327
5     142
6     115

0    20115
5    13064
3    12336
6    12190
4    11772
1    11130
2    10990




### Must-have feature

- Understand the requirements from the client.
- Create a product that answers to their needs.

### Nice-to-have
- Deployment is done using docker and Heroku

### Sample Project Pipeline
1. Start downloading the files
2. Analyze your folder contents; create a train, validation and test set.
3. Do proper preprocessing steps using OpenCV (which steps make sense, which don't, discuss these with your colleagues)
4. Import and existing model from the [keras models](https://keras.io/api/applications/)
5. Adjust (or add) the output layer to only contain the number of classes you are interested in.
6. After building your model, it's time to train them. For a pre-existing model, you might want to first train your added layers by freezing the existing layers. Use the [transfer learning guide](https://keras.io/guides/transfer_learning/) to help you set up an efficient workflow. 
7. You will use the training set to train, after which you will use your validation set to see what the actual accuracy might be. If it is not sufficient, you will need to adjust your model hyper-parameters (learning rate, number of layers, nodes,...) and train again. Depending on your transfer learning set-up, your training time might be minimal, and you can test more hyper-parameter options (random search, grid search...)
8. Once your model accuracy is sufficient (you choose what you deem sufficient), you can test on your testing set. You do not use your test set to further tune your hyper-parameters
9. Plot all the relevant evaluation metrics for your model (confusion matrix, ROC, precision/recall, validation accuracy,...)
10. Discuss which evaluation metrics are the most important for this challenge
11. Deploy an application with Flask where the customer can upload an image and get the model prediction


### Tips

- To have a good accuracy you will need to preprocess your images. OpenCV is amazing for that.

### Presentation

We will randomly select 10 presenters to showcase their project in front of the group.

- You have to make a nice presentation **with a professional design**.
- You have **10 minutes** to present (with Q&A). **You can't use more time**, **you can't use less time**.
- You **CAN'T show code or jupyter notebook** during the presentation.
- Remember, you present a SaaS (**S**oftware **A**s **A** **S**ervice), so present it like it is.


## Deliverables

1. Pimp up the README file:

   - Description
   - Installation
   - Usage
   - (Visuals)
   - (Contributors)
   - (Timeline)
   - (Personal situation)
   - (Pending things to do)

2. Present your results in front of the group in **10mins max**.
3. Your fully working proof of concept.

### Steps

1. Create the repository
2. Study the request (What & Why ?)
3. Identify technical challenges (How ?)
4. Create the requested product.
5. Deploy the product.

## Evaluation criteria

| Criteria       | Indicator                                                            | Yes/No |
| -------------- | -------------------------------------------------------------------- | ------ |
| 1. Is complete | Your product answers to all the client's demands.                    | [ ]    |
|                | Your README is detailed as expected.                                 | [ ]    |
|                | One is able to upload a picture to the website and have it analyzed. | [ ]    |
|                | The website returns a prediction along with its accuracy.            | [ ]    |
|                | You present the product well.                                        | [ ]    |
| 2. Is good     | The repo doesn't contain unnecessary files.                          | [ ]    |
|                | You used typing.                                                     | [ ]    |
|                | The presentation is clean.                                           | [ ]    |
|                | Your code is clean and commented.                                    | [ ]    |
|                | Your product is easy to use.                                         | [ ]    |
|                | Your communication with the client was clear.                        | [ ]    |

## Quotes

“Everyone should have health insurance? I say everyone should have health care. I'm not selling insurance.”
_- Dennis Kucinich_

![You've got this!](https://media.giphy.com/media/eLujANBginsZl8PZZL/giphy.gif)