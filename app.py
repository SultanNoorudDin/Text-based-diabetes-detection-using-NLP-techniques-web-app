from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load and preprocess the data
diabetes_dataset = pd.read_csv('diabetes.csv')
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Flask app setup
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    input_data = [
        float(request.form['pregnancies']),
        float(request.form['glucose']),
        float(request.form['blood_pressure']),
        float(request.form['skin_thickness']),
        float(request.form['insulin']),
        float(request.form['bmi']),
        float(request.form['dpf']),
        float(request.form['age']),
    ]
    
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)

    prediction = classifier.predict(std_data)

    result = "The person is diabetic" if prediction[0] == 1 else "The person is not diabetic"
    return render_template('result.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
