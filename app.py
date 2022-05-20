import os
import numpy as np
from util import base64_to_pil

from flask import Flask, request, render_template,  jsonify, redirect
import tensorflow as tf
from tensorflow.python.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

#model for benign or malignant
Model_json = "./model/modelbm.json"
Model_weigths = "./model/modelbm.h5"

#model for mole class
Modelmk_json = "./model/modelmk.json"
Modelmk_weigths = "./model/modelmk.h5"

labelsmk = {
    0: 'Actinic keratoses (akiec)',
    1: 'Basal cell carcinoma (bcc)',
    2: 'Benign keratosis-like lesions (bkl)',
    3: 'Dermatofibroma (df)',
    4: 'Melanoma (mel)',
    5: 'Melanocytic nevi (nv)',
    6: 'Vascular lesions (vasc)',
}

labels = {
    0: 'benign',
    1: 'malignant'
}


app = Flask(__name__)

def get_MoleClassifierModel(modeljson,weights):
    '''
    Function to load saved model and weights 
    '''
    model_json = open(modeljson, 'r')
    loaded_model_json = model_json.read()
    model_json.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights)
    return model  


def model_predict(img:image, model,dima:int,dimb:int):
    '''
    Get the image data and return predictions
    '''
    img = img.resize((dima, dimb))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds



@app.route('/', methods=['GET'])
def index():
    '''
    main page
    '''
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    '''
    predict function, makes the prediction and returns highest predicted in json format
    '''
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)
        
        # initialize models
        model = get_MoleClassifierModel(Model_json,Model_weigths)
        modelmk=get_MoleClassifierModel(Modelmk_json,Modelmk_weigths)

        # Make predictions
        preds = model_predict(img, model,120,90)
        predsmk=model_predict(img,modelmk,32,32)
        
        preds=preds[0][0]
        predsmk=np.argmax(predsmk)

        if preds<0.5:
            preds=1-preds

        predsmk=labelsmk.get(predsmk) 
               
        max_index_col = np.round(preds,0)
           
        pred_probabilty = " % {:.2f}".format(preds*100)
        preds=np.round(preds,0)
                              
        
        result=labels.get(max_index_col) + pred_probabilty + "-" + predsmk

        return jsonify(result=result )
    return None


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)