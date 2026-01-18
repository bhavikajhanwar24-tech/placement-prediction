import os
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

print("Current directory:", os.getcwd())

# Load your model.pkl — make sure model.pkl is in the same folder as app.py
import joblib

model = joblib.load('placement_model.pkl')

print("Model loaded successfully")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    cgpa_raw = request.form.get('cgpa')
    iq_raw = request.form.get('iq')

    print(f" CGPA: {cgpa_raw}, Raw inputs received - IQ: {iq_raw}")

    try:
        cgpa = float(cgpa_raw)
        iq = float(iq_raw)
    except Exception as e:
        print(f"Error parsing inputs: {e}")
        return "Invalid input! Please enter valid numbers for IQ and CGPA.", 400

    print(f"Parsed inputs - CGPA: {cgpa},IQ: {iq}")

    prediction = model.predict([[cgpa, iq]])
    print(f"Raw prediction: {prediction}, type: {type(prediction)}")

    pred_value = prediction[0]
    print(f"Prediction value: {pred_value}, type: {type(pred_value)}")

    if str(pred_value) == '1' or pred_value == 1:
        prediction_text = "Placement Done"
    else:
        prediction_text = "Placement Not Done"

    print(f"Final prediction text: {prediction_text}")

    return render_template('result.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
