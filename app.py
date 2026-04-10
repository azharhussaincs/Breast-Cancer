import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os
from sklearn.datasets import load_breast_cancer

# Page Config
st.set_page_config(
    page_title="Breast Cancer Diagnostics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Professional Look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
    }
    .prediction-card {
        padding: 2rem;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Load artifacts
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('artifacts/best_model.joblib')
        scaler = joblib.load('artifacts/scaler.joblib')
        data = load_breast_cancer()
        return model, scaler, data
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None, None

model, scaler, data = load_artifacts()

if model is None:
    st.stop()

# Header Section
st.title("🩺 Breast Cancer Diagnostic Assistant")
st.markdown("""
    Welcome to the **Professional Breast Cancer Diagnostic Tool**. This application uses a high-accuracy Logistic Regression model 
    to assist in the classification of breast masses as **Malignant** or **Benign** based on FNA digitized image features.
""")

# Sidebar Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home & Prediction", "Model Performance", "EDA Visualizations"])

if page == "Home & Prediction":
    # Organize Features into Categories
    feature_names = data.feature_names
    categories = {
        "Mean Features": [f for f in feature_names if "mean" in f],
        "Standard Error Features": [f for f in feature_names if "error" in f],
        "Worst Features": [f for f in feature_names if "worst" in f]
    }

    st.subheader("Patient Diagnostic Parameters")
    st.info("Adjust the sliders in the tabs below to match the patient's diagnostic data.")

    # Input Tabs
    tab1, tab2, tab3 = st.tabs(["Mean", "Standard Error", "Worst"])
    
    inputs = {}
    means = data.data.mean(axis=0)
    mins = data.data.min(axis=0)
    maxs = data.data.max(axis=0)

    with tab1:
        cols = st.columns(2)
        for i, f_name in enumerate(categories["Mean Features"]):
            idx = list(feature_names).index(f_name)
            inputs[f_name] = cols[i % 2].slider(f_name.replace("mean", "").strip().title(), float(mins[idx]), float(maxs[idx]), float(means[idx]))

    with tab2:
        cols = st.columns(2)
        for i, f_name in enumerate(categories["Standard Error Features"]):
            idx = list(feature_names).index(f_name)
            inputs[f_name] = cols[i % 2].slider(f_name.replace("error", "").strip().title(), float(mins[idx]), float(maxs[idx]), float(means[idx]))

    with tab3:
        cols = st.columns(2)
        for i, f_name in enumerate(categories["Worst Features"]):
            idx = list(feature_names).index(f_name)
            inputs[f_name] = cols[i % 2].slider(f_name.replace("worst", "").strip().title(), float(mins[idx]), float(maxs[idx]), float(means[idx]))

    input_df = pd.DataFrame([inputs])

    st.markdown("---")
    
    col_res, col_prob = st.columns([1, 1])

    with col_res:
        st.subheader("Run Analysis")
        if st.button("Predict Diagnosis"):
            # Preprocess
            input_scaled = scaler.transform(input_df[feature_names]) # Ensure correct order
            
            # Predict
            prediction = model.predict(input_scaled)
            prob = model.predict_proba(input_scaled)[0]
            
            # Display Result
            result = "Benign" if prediction[0] == 1 else "Malignant"
            color = "#28a745" if result == "Benign" else "#dc3545"
            
            st.markdown(f"""
                <div class="prediction-card">
                    <h2 style="color: {color};">{result}</h2>
                    <p>Based on the provided parameters, the model predicts a <b>{result}</b> tumor.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_prob:
                st.subheader("Probability Distribution")
                fig = go.Figure(go.Bar(
                    x=['Malignant', 'Benign'],
                    y=prob,
                    marker_color=['#dc3545', '#28a745']
                ))
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, width="stretch")

elif page == "Model Performance":
    st.header("Model Evaluation Metrics")
    st.write("The current best model is a **Logistic Regression** model optimized with GridSearchCV.")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "97.4%")
    col2.metric("F1-Score", "0.98")
    col3.metric("Precision", "0.97")
    col4.metric("Recall", "0.99")
    
    st.subheader("Confusion Matrix")
    if os.path.exists('artifacts/cm_Logistic_Regression.png'):
        st.image('artifacts/cm_Logistic_Regression.png', caption="Confusion Matrix for Logistic Regression")
    else:
        st.warning("Confusion matrix artifact not found. Please run ml_pipeline.py first.")

elif page == "EDA Visualizations":
    st.header("Exploratory Data Analysis")
    st.write("Visual insights from the training dataset used to develop the model.")
    
    viz_choice = st.selectbox("Select Visualization", ["PCA Plot", "Correlation Heatmap", "Histograms", "Boxplots"])
    
    if viz_choice == "PCA Plot":
        st.image('artifacts/pca_plot.png', width="stretch")
    elif viz_choice == "Correlation Heatmap":
        st.image('artifacts/correlation_heatmap.png', width="stretch")
    elif viz_choice == "Histograms":
        st.image('artifacts/histograms.png', width="stretch")
    elif viz_choice == "Boxplots":
        st.image('artifacts/boxplot.png', width="stretch")

st.sidebar.markdown("---")
st.sidebar.info("Developed for Machine Learning Assignment #2. Dataset: Wisconsin Breast Cancer (Diagnostic).")
