import os
import numpy as np
from util import base64_to_pil
from itertools import islice


from flask import Flask, request, render_template,  jsonify, redirect
import tensorflow as tf
from tensorflow.python.keras.models import model_from_json
from tensorflow.keras.preprocessing import image



#model for mole class
Modelmk_json = "./model/modelmk.json"
Modelmk_weigths = "./model/modelmk.h5"

labelsmk = {
    0: 'Actinic keratoses',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis-like lesions',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic nevi',
    6: 'Vascular lesions',
}

labels = {
    0: 'benign',
    1: 'malignant'
}


app = Flask(__name__)

def take(n, iterable):
    "Return first n items of the iterable as a dict"
    return dict(islice(iterable, n))


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
    x=x/255
    #print(x)
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
        #model = get_MoleClassifierModel(Model_json,Model_weigths)
        modelmk=get_MoleClassifierModel(Modelmk_json,Modelmk_weigths)

        # Make predictions
        predsmk=model_predict(img,modelmk,32,32)
        predsmk= np.around(predsmk*100,0)
        labelsof_type = list(labelsmk.values())
        predictions = predsmk.tolist()


        dict1 = dict(zip(labelsof_type, np.around(predictions[0],2)))
                  
        
        #sort the predictions descending
        sorted_predictions = {}
        sorted_keys = sorted(dict1, key=dict1.get,reverse=True) 

        for w in sorted_keys:
            sorted_predictions[w] = dict1[w]
        
        for key, value in sorted_predictions.items():
            sorted_predictions[key] = '% ' + str(value)
        
        top2_predictions = take(2, sorted_predictions.items())

        predictions= top2_predictions
 

        it_values = iter(predictions.values())    
        firstv, secondv = next(it_values), next(it_values)    
        it_keys = iter(predictions.keys())
        firstk, secondk = next(it_keys), next(it_keys)
        
        predictions= firstv + " " + firstk
        predresult=secondv +" " + secondk
        

        

        return jsonify(result=predictions,predresult=predresult)
    return None


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)