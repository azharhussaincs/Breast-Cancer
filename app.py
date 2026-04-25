# Welcome to the Heart of the App! 
# We're using Streamlit to create a professional, interactive dashboard.

import streamlit as st  # The main framework for our web app
import pandas as pd     # For handling input data as tables
import numpy as np      # For numerical computations
import joblib           # To load our pre-trained model and scaler
import plotly.graph_objects as go  # For interactive, beautiful charts
import os               # To check if files exist before we try to open them
from sklearn.datasets import load_breast_cancer  # To get the feature names and metadata

# Page Config: Setting up the browser tab title and layout
st.set_page_config(
    page_title="Breast Cancer Diagnostics",
    page_icon="🎗️",
    layout="wide",  # Use the full width of the screen
    initial_sidebar_state="expanded", # Keep the menu open by default
)

# Custom CSS: This is where we make the app look like a professional medical tool.
# We're using custom fonts, gradients, and polished card designs.
st.markdown("""
    <style>
    /* Importing the 'Inter' font for a clean, modern look */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Soft background gradient for the app */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Styling the Sidebar */
    div[data-testid="stSidebarNav"] {
        background-color: white;
        border-radius: 0 20px 20px 0;
        box-shadow: 2px 0 10px rgba(0,0,0,0.05);
    }
    
    /* Making the Tabs look like modern buttons */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #ffffff;
        padding: 10px 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 10px 25px;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
        font-weight: 600;
        color: #475569;
    }
    
    /* Hover effect for tabs */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f1f5f9;
        border-color: #cbd5e1;
    }
    
    /* Active tab styling (blue gradient) */
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    
    /* Cards for our sliders and results */
    .slider-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        border: 1px solid #f1f5f9;
        margin-bottom: 15px;
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
    
    /* Green left border for Benign, Red for Malignant */
    .benign { border-left-color: #28a745; }
    .malignant { border-left-color: #dc3545; }
    
    /* Main Prediction Button Styling */
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
    }
    
    /* Small info cards for performance metrics */
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

# Helper function to load our saved brain (model) and scaling rules
@st.cache_resource  # We cache this so it only runs once and stays fast
def load_artifacts():
    try:
        # Loading the best model we found during training
        model = joblib.load('artifacts/best_model.joblib')
        # Loading the scaler to make sure user inputs are normalized correctly
        scaler = joblib.load('artifacts/scaler.joblib')
        # Also need the dataset metadata for feature names
        data = load_breast_cancer()
        return model, scaler, data
    except Exception as e:
        # If something goes wrong (like missing files), we show an error
        st.error(f"Error loading model or scaler: {e}")
        return None, None, None

# Run the loader
model, scaler, data = load_artifacts()

# If the model didn't load, we can't do anything, so we stop
if model is None:
    st.stop()

# --- Main App Header ---
st.markdown('<h1 style="color: #1e3a8a; font-size: 2.5rem; font-weight: 800; margin-bottom: 0;">🎗️ Breast Cancer Diagnostic Assistant</h1>', unsafe_allow_html=True)
st.markdown("""
    <p style="font-size: 1.1rem; color: #4b5563; margin-bottom: 2rem;">
        Professional clinical tool for breast mass classification using high-precision machine learning.
    </p>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home & Prediction", "Model Performance", "EDA Visualizations"])

# Page 1: The Prediction Tool
if page == "Home & Prediction":
    feature_names = data.feature_names
    # Grouping features into logical sections so they're easier for a doctor to read
    categories = {
        "Mean Features": [f for f in feature_names if "mean" in f],
        "Standard Error Features": [f for f in feature_names if "error" in f],
        "Worst Features": [f for f in feature_names if "worst" in f]
    }

    st.subheader("Patient Diagnostic Parameters")
    st.info("Adjust the sliders in the tabs below to match the patient's diagnostic data.")

    # Organized tabs for the 30 different sliders
    tab1, tab2, tab3 = st.tabs(["📊 Mean Features", "📉 Standard Error", "🔍 Worst Case (Max)"])
    
    inputs = {}
    # Getting some stats so our sliders have sensible default values
    means = data.data.mean(axis=0)
    mins = data.data.min(axis=0)
    maxs = data.data.max(axis=0)

    # --- Tab 1: Means ---
    with tab1:
        st.markdown('<div class="slider-card">', unsafe_allow_html=True)
        st.markdown('<h4 style="color: #334155; margin-top: 0;">Mean Values of Cell Nuclei</h4>', unsafe_allow_html=True)
        cols = st.columns(2) # Two-column layout for better use of space
        for i, f_name in enumerate(categories["Mean Features"]):
            idx = list(feature_names).index(f_name)
            display_name = f_name.replace("mean", "").strip().title()
            # Creating a slider for each feature in this category
            inputs[f_name] = cols[i % 2].slider(f"📏 {display_name}", float(mins[idx]), float(maxs[idx]), float(means[idx]), help=f"Average {display_name.lower()} across cell nuclei")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Tab 2: Standard Errors ---
    with tab2:
        st.markdown('<div class="slider-card">', unsafe_allow_html=True)
        st.markdown('<h4 style="color: #334155; margin-top: 0;">Standard Error of Measurements</h4>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, f_name in enumerate(categories["Standard Error Features"]):
            idx = list(feature_names).index(f_name)
            display_name = f_name.replace("error", "").strip().title()
            inputs[f_name] = cols[i % 2].slider(f"⚖️ {display_name}", float(mins[idx]), float(maxs[idx]), float(means[idx]), help=f"Standard error of {display_name.lower()}")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Tab 3: "Worst" Case Values ---
    with tab3:
        st.markdown('<div class="slider-card">', unsafe_allow_html=True)
        st.markdown('<h4 style="color: #334155; margin-top: 0;">"Worst" (Largest) Case Measurements</h4>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, f_name in enumerate(categories["Worst Features"]):
            idx = list(feature_names).index(f_name)
            display_name = f_name.replace("worst", "").strip().title()
            inputs[f_name] = cols[i % 2].slider(f"🔺 {display_name}", float(mins[idx]), float(maxs[idx]), float(means[idx]), help=f"Largest {display_name.lower()} measured")
        st.markdown('</div>', unsafe_allow_html=True)

    # Put all slider values into a DataFrame so the model can read it
    input_df = pd.DataFrame([inputs])

    st.markdown("---")
    
    # Bottom section for results
    col_res, col_prob = st.columns([1, 1])

    with col_res:
        st.subheader("Run Analysis")
        # Only predict when the user clicks the button
        if st.button("Predict Diagnosis"):
            # Step 1: Scale the inputs exactly how the training data was scaled
            input_scaled = scaler.transform(input_df[feature_names]) 
            
            # Step 2: Ask the model for its prediction
            prediction = model.predict(input_scaled)
            # And ask how confident it is (probabilities)
            prob = model.predict_proba(input_scaled)[0]
            
            # Step 3: Interpret the result for humans
            result = "Benign" if prediction[0] == 1 else "Malignant"
            color = "#28a745" if result == "Benign" else "#dc3545"
            card_class = "benign" if result == "Benign" else "malignant"
            icon = "✅" if result == "Benign" else "⚠️"
            
            # Display the result in a big, clear card
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
            
            # Show the probability distribution as a bar chart
            with col_prob:
                st.subheader("Probability Distribution")
                fig = go.Figure(go.Bar(
                    x=['Malignant', 'Benign'],
                    y=prob,
                    marker_color=['#dc3545', '#28a745'] # Red for bad, Green for good
                ))
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)

# Page 2: Performance Metrics
elif page == "Model Performance":
    st.markdown('<h1 class="section-header">Model Evaluation Metrics</h1>', unsafe_allow_html=True)
    st.write("The current best model is a **Logistic Regression** model optimized with GridSearchCV.")
    
    # Display the big four metrics in polished cards
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
    # Show the confusion matrix image generated by our pipeline
    if os.path.exists('artifacts/cm_Logistic_Regression.png'):
        st.image('artifacts/cm_Logistic_Regression.png', caption="Confusion Matrix for Logistic Regression")
    else:
        st.warning("Confusion matrix artifact not found. Please run ml_pipeline.py first.")

# Page 3: Visual Insights (EDA)
elif page == "EDA Visualizations":
    st.markdown('<h1 class="section-header">Exploratory Data Analysis</h1>', unsafe_allow_html=True)
    st.write("Visual insights from the training dataset used to develop the model.")
    
    # User can pick which chart they want to see
    viz_choice = st.selectbox("Select Visualization", ["PCA Plot", "Correlation Heatmap", "Histograms", "Boxplots"])
    
    st.markdown('<div class="prediction-card" style="padding: 1rem; border-left: none;">', unsafe_allow_html=True)
    # Load the requested image from the artifacts folder
    if viz_choice == "PCA Plot":
        st.image('artifacts/pca_plot.png', use_container_width=True)
        st.info("PCA shows how the data points are distributed in a 2D space, helping identify clusters.")
    elif viz_choice == "Correlation Heatmap":
        st.image('artifacts/correlation_heatmap.png', use_container_width=True)
        st.info("The heatmap shows relationships between features. Highly correlated features can often be simplified.")
    elif viz_choice == "Histograms":
        st.image('artifacts/histograms.png', use_container_width=True)
        st.info("Histograms show the frequency distribution of individual features.")
    elif viz_choice == "Boxplots":
        st.image('artifacts/boxplot.png', use_container_width=True)
        st.info("Boxplots are useful for identifying outliers and understanding the spread of data.")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer info
st.sidebar.markdown("---")
st.sidebar.info("Developed for Machine Learning Assignment #2. Dataset: Wisconsin Breast Cancer (Diagnostic).")
