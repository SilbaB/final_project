import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the trained model
model = joblib.load("best_model.pkl")

st.title("Heart Disease Prediction App")


age = st.number_input("Age", min_value=29, max_value=77, value=30, step=1)
sex= st.selectbox("sex", [0, 1])
cp = st.selectbox("cp", [1, 2, 3, 4])
trestbps = st.number_input("trestbps", min_value=94, max_value=200, value=100, step=1)
chol = st.number_input("chol", min_value=126, max_value=564, value=300, step=1)
fbs= st.selectbox("fbs", [0, 1])
restecg= st.selectbox("restecg", [0, 1])
thalach= st.number_input("thalach", min_value=71, max_value=202, value=130, step=1)
exang= st.selectbox("exang", [0, 1])
oldpeak= st.selectbox("oldpeak", [0, 1,2,3,4,5,6])
slope= st.selectbox("fbs", [0, 1, 2, 3])
ca= st.selectbox("ca", [0, 1, 2, 3])
thal= st.selectbox("thal", [3, 4, 5, 6, 7])
# present= st.selectbox("present", [0, 1])

if st.button("Predict"):
    # Process input values
    inputed = pd.DataFrame(
        {
            "age": [age],
            "sex": [sex],
            "cp": [cp],
            "trestbps": [trestbps],
            "chol": [chol],
            "fbs": [fbs],
            "restecg": [restecg],
            "thalach": [thalach],
            "exang": [exang],
            "oldpeak": [oldpeak],
            "slope": [slope],
            "ca": [ca],
            "thal": [thal],
            # "present": [present],

          
        }
    )

    # Scale input data using the same scaler used during training
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(inputed)

    # Make a prediction using the trained model
    prediction = model.predict(input_data_scaled)

      # Display the prediction
    if prediction[0] == 1:
        st.success("The patient is at risk of heart disease.")
    else:
        st.success("The patient is not at risk of heart disease.")
