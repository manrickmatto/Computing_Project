from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
with open("Swell.pkl", "rb") as file:
    model = pickle.load(file)

mapping_dict = {'Stressed': {0: 'You seem okay', 1: 'You seem a little stressed'}}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        features = [
            int(request.form['HR']),
            #int(request.form['BVP']),
        ]
        prediction = model.predict([features])[0]
        predicted_Stress = mapping_dict['Stressed'][prediction]

        return render_template('index.html', prediction_text=f"Prediction: {predicted_Stress}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__": 
    app.run(debug=True)