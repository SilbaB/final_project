import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the trained model
model = joblib.load("best_model.pkl")

st.title("Heart Disease Prediction App")
st.write("Age ofnthe patient in years")
st.write("sex: Sex of the patient,0:Female,1:Male")
st.write("cp,Chest pain type. 0:Typical angina (chest pain related to decreased blood supply to the heart),1:Atypical angina (chest pain not related to heart),2:Non-anginal pain (pain not related to the heart),3: Asymptomatic (no chest pain)")
st.write("trestbps: Resting blood pressure (in mm Hg) on admission to the hospital.")
st.write("chol: Serum cholesterol in mg/dl.")
st.write("bs: Fasting blood sugar > 120 mg/dl.,0:=False,1:True")
st.write("restecg: Resting electrocardiographic results.0: Normal,1:Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV),2:Showing probable or definite left ventricular hypertrophy by Estes' criteria")
st.write("thalach: Maximum heart rate achieved during exercise.")
st.write("exang: Exercise-induced angina (chest pain induced by exercise).0=No,1=Yes")
st.write("oldpeak: ST depression induced by exercise relative to rest (a measure of abnormal heart activity).")
st.write("slope: The slope of the peak exercise ST segment.0: Upsloping (better heart rate prognosis),1: Flat (worse heart rate prognosis),2: Downsloping (poorest heart rate prognosis),")
st.write("ca: Number of major vessels (0-3) colored by fluoroscopy (a procedure to visualize blood flow through coronary arteries).")
st.write("thal: Thalassemia (a blood disorder involving lower-than-normal oxygen-carrying protein).,0: Null (not used in some versions of the dataset),1: Fixed defect (no blood flow in some part of the heart),2: Normal,3: Reversible defect (a blood flow is normal under resting conditions but abnormal during exercise)")
st.write("target: Diagnosis of heart disease (the predicted attribute).,0 = No heart disease,1 = Heart disease present")

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
        st.success("No heart disease.")
    else:
        st.success("Heart disease Present.")
