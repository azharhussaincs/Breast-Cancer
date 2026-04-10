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
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Professional Look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    div[data-testid="stSidebarNav"] {
        background-color: white;
        border-radius: 0 20px 20px 0;
        box-shadow: 2px 0 10px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 10px 10px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        border: 1px solid #e1e4e8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #007bff !important;
        color: white !important;
    }
    .prediction-card {
        padding: 2.5rem;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-left: 10px solid;
        margin-top: 1rem;
    }
    .benign { border-left-color: #28a745; }
    .malignant { border-left-color: #dc3545; }
    
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        font-weight: 700;
        border: none;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        border-bottom: 4px solid #007bff;
    }
    .section-header {
        color: #1e3a8a;
        font-weight: 700;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.5rem;
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
st.markdown('<h1 style="color: #1e3a8a; font-size: 2.5rem; font-weight: 800; margin-bottom: 0;">🎗️ Breast Cancer Diagnostic Assistant</h1>', unsafe_allow_html=True)
st.markdown("""
    <p style="font-size: 1.1rem; color: #4b5563; margin-bottom: 2rem;">
        Professional clinical tool for breast mass classification using high-precision machine learning.
    </p>
""", unsafe_allow_html=True)

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
            card_class = "benign" if result == "Benign" else "malignant"
            icon = "✅" if result == "Benign" else "⚠️"
            
            st.markdown(f"""
                <div class="prediction-card {card_class}">
                    <h1 style="color: {color}; margin-top: 0;">{icon} {result}</h1>
                    <p style="font-size: 1.2rem; color: #4b5563;">
                        The model predicts a <b>{result}</b> tumor with <b>{prob[prediction[0]]*100:.1f}%</b> confidence.
                    </p>
                    <hr style="border: 0; border-top: 1px solid #e5e7eb; margin: 1.5rem 0;">
                    <p style="font-size: 0.9rem; color: #6b7280; font-style: italic;">
                        * This is an AI-assisted tool and should not replace professional medical diagnosis.
                    </p>
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
    st.markdown('<h1 class="section-header">Model Evaluation Metrics</h1>', unsafe_allow_html=True)
    st.write("The current best model is a **Logistic Regression** model optimized with GridSearchCV.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h3>97.4%</h3><p>Accuracy</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>0.98</h3><p>F1-Score</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>0.97</h3><p>Precision</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h3>0.99</h3><p>Recall</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("Confusion Matrix")
    if os.path.exists('artifacts/cm_Logistic_Regression.png'):
        st.image('artifacts/cm_Logistic_Regression.png', caption="Confusion Matrix for Logistic Regression")
    else:
        st.warning("Confusion matrix artifact not found. Please run ml_pipeline.py first.")

elif page == "EDA Visualizations":
    st.markdown('<h1 class="section-header">Exploratory Data Analysis</h1>', unsafe_allow_html=True)
    st.write("Visual insights from the training dataset used to develop the model.")
    
    viz_choice = st.selectbox("Select Visualization", ["PCA Plot", "Correlation Heatmap", "Histograms", "Boxplots"])
    
    st.markdown('<div class="prediction-card" style="padding: 1rem; border-left: none;">', unsafe_allow_html=True)
    if viz_choice == "PCA Plot":
        st.image('artifacts/pca_plot.png', width="stretch")
        st.info("PCA shows how the data points are distributed in a 2D space, helping identify clusters.")
    elif viz_choice == "Correlation Heatmap":
        st.image('artifacts/correlation_heatmap.png', width="stretch")
        st.info("The heatmap shows relationships between features. Highly correlated features can often be simplified.")
    elif viz_choice == "Histograms":
        st.image('artifacts/histograms.png', width="stretch")
        st.info("Histograms show the frequency distribution of individual features.")
    elif viz_choice == "Boxplots":
        st.image('artifacts/boxplot.png', width="stretch")
        st.info("Boxplots are useful for identifying outliers and understanding the spread of data.")
    st.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.info("Developed for Machine Learning Assignment #2. Dataset: Wisconsin Breast Cancer (Diagnostic).")
