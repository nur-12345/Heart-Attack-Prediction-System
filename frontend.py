import streamlit as st
import requests
import numpy as np

st.title("❤️ Heart Attack Prediction")

# Input
pressurehight = st.number_input("Systolic BP (pressurehight)", value=120.0)
pressurelow = st.number_input("Diastolic BP (pressurelow)", value=80.0)
pulse_pressure = pressurehight - pressurelow
bp_ratio = pressurehight / pressurelow
glucose = st.number_input("Glucose Level", value=130.0)
troponin = st.number_input("Troponin Level", value=0.3)
kcm = st.number_input("KCM Level", value=5.0)
impluse = st.number_input("Pulse (impluse)", value=70.0)
gender = st.selectbox("Gender", ["Female", "Male"])
gender_encoded = 1 if gender == "Male" else 0

# Derived features
glucose_log = np.log1p(glucose)
troponin_log = np.log1p(troponin)
kcm_log = np.log1p(kcm)
impluse_log = np.log1p(impluse)

high_glucose_flag = int(glucose > 140)
high_troponin = int(troponin > 0.4)
high_kcm = int(kcm > 6.0)
glucose_troponin = glucose * troponin
bp_glucose_ratio = pressurehight / (glucose + 1)

# Submit
if st.button("Predict"):
    payload = {
        "pressurehight": pressurehight,
        "pressurelow": pressurelow,
        "pulse_pressure": pulse_pressure,
        "bp_ratio": bp_ratio,
        "glucose_troponin": glucose_troponin,
        "bp_glucose_ratio": bp_glucose_ratio,
        "glucose_log": glucose_log,
        "impluse_log": impluse_log,
        "kcm_log": kcm_log,
        "troponin_log": troponin_log,
        "high_glucose_flag": high_glucose_flag,
        "high_troponin": high_troponin,
        "high_kcm": high_kcm,
        "gender": gender_encoded
    }

    response = requests.post("http://localhost:8000/predict", json=payload)
    if response.status_code == 200:
        result = response.json()
        if "Positive" in result["prediction"]:
            st.error("⚠️ Risk of Heart Attack Detected!")
        else:
            st.success("✅ Low Risk of Heart Attack.")
    else:
        st.error("❌ Something went wrong with the prediction request.")
