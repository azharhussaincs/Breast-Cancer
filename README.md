# Breast Cancer Diagnostic Assistant

This project is a professional clinical tool for breast mass classification using high-precision machine learning. It features a Streamlit-based web application for real-time diagnostic assistance.

## 🚀 Live Demo
The application is ready for deployment on [Streamlit Community Cloud](https://share.streamlit.io/).

## 🛠️ Features
- **Real-time Prediction**: Classify tumors as Benign or Malignant.
- **Model Performance**: Visual metrics including Confusion Matrix and F1-Score.
- **Exploratory Data Analysis (EDA)**: Interactive visualizations like PCA plots and Correlation Heatmaps.

## 📦 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/azharhussaincs/Breast-Cancer.git
   cd Breast-Cancer
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training pipeline (optional, artifacts are included):
   ```bash
   python ml_pipeline.py
   ```
4. Launch the app:
   ```bash
   streamlit run app.py
   ```

## 📂 Project Structure
- `app.py`: Streamlit web application.
- `ml_pipeline.py`: Data preprocessing, model training, and evaluation.
- `artifacts/`: Saved model, scaler, and visualization plots.
- `.github/workflows/Depy.yml`: CI/CD pipeline for GitHub Actions.

## 🧪 Deployment
This project includes a `Depy.yml` GitHub Actions workflow that automatically verifies the code and runs the ML pipeline on every push to the `main` branch.
