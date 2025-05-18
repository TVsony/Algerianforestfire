from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# मॉडल और स्केलर लोड करना
try:
    ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
    standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))
except Exception as e:
    print("Model loading error:", e)

# होमपेज के लिए रूट
@app.route('/')
def index():
    return render_template('home.html')  # HTML form दिखाएगा

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))

            # Class और Region को numeric में कनवर्ट करें
            class_map = {'fire': 1.0, 'not fire': 0.0}
            region_map = {'north': 1.0, 'south': 0.0}

            Classes = class_map.get(request.form.get('Classes').lower(), 0.0)
            Region = region_map.get(request.form.get('Region').lower(), 0.0)

            input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
            new_data_scaled = standard_scaler.transform(input_data)
            result = ridge_model.predict(new_data_scaled)

            return render_template('home.html', result=round(result[0], 2))
        except Exception as e:
            return render_template('home.html', result=f"Error in prediction: {str(e)}")
    else:
        return render_template('home.html', result=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)


