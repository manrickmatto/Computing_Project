from flask import Flask, render_template, request
import pickle
import numpy as np
import joblib
import pandas as pd



app = Flask(__name__)
with open("Swell.pkl", "rb") as file:
    model = pickle.load(file)

HR_model = joblib.load("Swell.pkl")
BVP_model = joblib.load("WESADBVP.pkl")
ecg_model = joblib.load("WESADECG.pkl")

mapping_dict = {'Stressed': {0: 'You seem okay', 1: 'You seem a little stressed'}}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['GET','POST'])
def result():
    if request.method == 'POST':
        print ("FORM DATA:", request.form)
    
        #Readings from the html form
        HR = request.form.get('HR')
        BVP = request.form.get('BVP')
        ecg = request.form.get('ECG')

        #Error display on submitting
        if HR is None or BVP is None or ecg is None:
            return "Please enter all values", 400
        try:
            HR_predict = float(HR)
            ecg_predict = float(ecg)
            BVP_predict = float(BVP)
        except ValueError:
            return "All values must be numbers", 400

        #HR model: convert to pandas
        HR_input = pd.DataFrame(
            [[HR_predict]],
              columns=['HR']
        )
        
        HR_answer = HR_model.predict(HR_input)[0]
        

        #BVP model
        BVP_input = pd.DataFrame(
            [[BVP_predict]],
            columns=['BVP']
        )
        BVP_answer = BVP_model.predict(BVP_input)[0]

        #ECG model
        ecg_input = pd.DataFrame(
            [[ecg_predict]],
            columns=['ecg']
        )
        ecg_answer = ecg_model.predict(ecg_input)[0]

        #get the prediction answer
        total_score = (BVP_answer + HR_answer + ecg_answer) / 3

        return render_template(
            'index.html',
            BVP_answer=BVP_answer,
            HR_answer=HR_answer,
            ECG_answer=ecg_answer,
            total_score=total_score
        )
    print(request.form)
    return render_template('index.html')

if __name__ == "__main__": 
    app.run(debug=True)


