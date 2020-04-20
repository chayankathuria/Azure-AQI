import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('rf.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
	
    if(output==0):
        flowr = 'Setosa'
    
    if(output==1):
        flowr = 'Versicolour'
        
    if(output==2):
        flowr = 'Virginica'
	
    return render_template('index.html', prediction_text='Iris Type should be: {}'.format(flowr))


if __name__ == "__main__":
    app.run(debug=True)
