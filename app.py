import os
import numpy as np
from util import base64_to_pil

from flask import Flask, request, render_template,  jsonify, redirect
import tensorflow as tf
from tensorflow.python.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

Model_json = "./model/model.json"
Model_weigths = "./model/model.h5"
labels = {
    0: 'Melanocytic nevi (nv)',
    1: 'Melanoma (mel)',
    2: 'Benign keratosis-like lesions (bkl)',
    3: 'Basal cell carcinoma (bcc)',
    4: 'Actinic keratoses (akiec)',
    5: 'Vascular lesions (vasc)',
    6: 'Dermatofibroma (df)'
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
   
    img = img.resize((28, 28))
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
        #max probable       
        pred_probabilty = " % {:.2f}".format(np.amax(preds)*100) 
        max_index_col = np.argmax(preds)
                    
        #get max probabal mole name
        result=labels.get(max_index_col) + pred_probabilty

        return jsonify(result=result )
    return None


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)