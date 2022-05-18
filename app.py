import os
import numpy as np
from util import base64_to_pil

from flask import Flask, request, render_template,  jsonify, redirect
import tensorflow as tf
from tensorflow.python.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

Model_json = "./model/modelbm.json"
Model_weigths = "./model/modelbm.h5"
labels = {
    0: 'benign',
    1: 'malignant'
}


app = Flask(__name__)

def get_MoleClassifierModel():
    '''
    Function to load saved model and weights
    '''
    model_json = open(Model_json, 'r')
    loaded_model_json = model_json.read()
    model_json.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(Model_weigths)
    return model  
    

def model_predict(img, model):
    '''
    Get the image data and return prediction
    '''
   
    img = img.resize((120, 90))
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
        
        # initialize model
        model = get_MoleClassifierModel()

        # Make prediction
        preds = model_predict(img, model)


        preds=preds[0][0]
        print(preds)
        max_index_col = np.round(preds,0)

        if preds<0.5:
            preds=1-preds
            

        pred_probabilty = " % {:.2f}".format(preds*100)
        preds=np.round(preds,0)
        
                       
        
        result=labels.get(max_index_col) + pred_probabilty

        return jsonify(result=result )
    return None


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)