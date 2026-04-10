import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.datasets import load_breast_cancer

# Load artifacts
try:
    model = joblib.load('artifacts/best_model.joblib')
    scaler = joblib.load('artifacts/scaler.joblib')
    data = load_breast_cancer()
    feature_names = data.feature_names
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

st.title("Breast Cancer Classification Web App")
st.write("""
This application uses a Machine Learning model (Logistic Regression) to predict whether a breast tumor is **Malignant** or **Benign** based on diagnostic features.
""")

st.sidebar.header("Input Features")

def user_input_features():
    inputs = {}
    # Use mean values as defaults for the sliders
    means = data.data.mean(axis=0)
    stds = data.data.std(axis=0)
    
    for i, name in enumerate(feature_names):
        # Create sliders for each feature
        inputs[name] = st.sidebar.slider(
            name, 
            float(data.data[:, i].min()), 
            float(data.data[:, i].max()), 
            float(means[i])
        )
    return pd.DataFrame([inputs])

input_df = user_input_features()

st.subheader("User Input Parameters")
st.write(input_df)

# Prediction
if st.button("Predict"):
    # Preprocess input
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    
    st.subheader("Prediction")
    result = "Malignant" if prediction[0] == 0 else "Benign"
    st.write(f"The model predicts the tumor is: **{result}**")
    
    st.subheader("Prediction Probability")
    st.write(pd.DataFrame(prediction_proba, columns=['Malignant', 'Benign']))

st.sidebar.markdown("---")
st.sidebar.write("Developed for Machine Learning Assignment #2")
