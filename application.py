import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application

ridge_model = pickle.load(open('model/ridger.pkl',"rb"))
standard_scaler = pickle.load(open('model/scaler.pkl',"rb"))


# Route for HomePage
@app.route('/')
def index():
    return render_template('index.html')
# Route for Prediction
@app.route('/predictdata', methods=["GET", "POST"])
def predict_datapoint():
    if request.method == 'POST':
        # Get form data from request
        Temperature = float(request.form['Temperature'])
        RH = float(request.form['RH'])
        Ws = float(request.form['Ws'])
        Rain = float(request.form['Rain'])
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        ISI = float(request.form['ISI'])
        Classes = int(request.form['Classes']) # Keep this as a string or map it to a number
        Region = int(request.form['Region'])   # Same for Region

        # Create input array for the model
        input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI,Classes,Region]])

        # Standardize the input data
        scaled_data = standard_scaler.transform(input_data)

        # Make prediction
        prediction = ridge_model.predict(scaled_data)

        # Return result in the template
        return render_template('home.html', result=prediction)

    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")