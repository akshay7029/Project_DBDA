from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open(r"C:\Users\gaura\Desktop\calories_model.pkl", "rb"))
scaler = pickle.load(open(r"C:\Users\gaura\Desktop\scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")  # No need to use full path

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    # Encode 'sex' field (male: 0, female: 1)
    sex = 0 if data['sex'] == 'male' else 1

    # Remaining inputs (excluding dropped ones)
    input_data = np.array([[
        sex,
        float(data['age']),
        float(data['height']),
        float(data['weight']),
        float(data['duration']),
        float(data['heart_rate']),
        float(data['body_temp'])
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    return render_template("index.html", prediction=f"{prediction:.2f} kcal")

if __name__ == '__main__':
    app.run(debug=True)
