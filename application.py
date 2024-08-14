from flask import Flask, request, app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd


application = Flask(__name__,static_url_path="/static")
app=application

#load piclke file
scaler=pickle.load(open("standardScalar.pkl", "rb"))
model = pickle.load(open("Prediction.pkl", "rb"))



## Route for prediction
@app.route('/',methods=['GET','POST'])
def predictdata():
    result = 0
    
    if request.method=='POST':
        gender=float(request.form.get("gender"))
        age = float(request.form.get('age'))
        currentsmoker = float(request.form.get('currentsmoker'))
        cigsperday = float(request.form.get('cigsperday'))
        bpmed=float(request.form.get("bpmed"))
        stroke = float(request.form.get('stroke'))
        prevelenthyp = float(request.form.get('prevelenthyp'))
        diabetes = float(request.form.get('diabetes'))
        totchol = float(request.form.get('totchol'))
        sysbp = float(request.form.get('sysbp'))
        diabp = float(request.form.get('diabp'))
        bmi = float(request.form.get('bmi'))
        heartrate = float(request.form.get('heartrate'))
        glucose = float(request.form.get('glucose'))

        new_data=scaler.transform([[
            gender,age,currentsmoker,cigsperday,bpmed,stroke,prevelenthyp,diabetes,totchol,
            sysbp,diabp,bmi,heartrate,glucose
        ]])

        predict=model.predict(new_data)
            
        return render_template('output.html',result = predict[0])

    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")