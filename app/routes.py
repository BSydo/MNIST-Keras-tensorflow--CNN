from flask import Flask,render_template,url_for,request
from app import app

import numpy as np
import base64

from controller import controller


#Initialize the useless part of the base64 encoded image.
init_Base64 = 21

model = controller()

#First route : Render the initial drawing template
@app.route('/')
def home():
	return render_template('draw.html')

#Second route : Use our model to make prediction - render the results page.
@app.route('/predict', methods=['POST'])
def predict():
    
    if request.method == 'POST':
        
        #Access the image
        draw = request.form['url']
        
        #Removing the useless part of the url.
        draw = draw[init_Base64:]
        
        #Decoding
        draw_decoded = base64.b64decode(draw)
        
        #Launch prediction
        my_prediction = model.predict_draw(draw_decoded)
        #Getting the index of the maximum prediction
        final_pred = np.argmax(my_prediction[0])
        
    return render_template('results.html', prediction =final_pred)