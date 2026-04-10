# Machine Learning Project Report: Breast Cancer Classification

## Phase 1: Dataset Selection & Problem Definition
- **Dataset**: Breast Cancer Wisconsin (Diagnostic) Dataset.
- **Introduction**: The dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.
- **Objective**: To develop a classification model that can predict whether a tumor is **Malignant** (0) or **Benign** (1) based on these features.
- **ML Type**: Supervised Learning (Binary Classification).

## Phase 2: Data Preprocessing & Exploratory Data Analysis (EDA)
- **Data Loading**: Loaded using `sklearn.datasets.load_breast_cancer`.
- **Handling Missing Values & Duplicates**: Checked for null values and duplicate entries (none were found in this standard dataset).
- **Outlier Detection**: Visualized outliers using boxplots for the first 10 features.
- **Feature Engineering**: Features were scaled using `StandardScaler` to normalize the data, ensuring that features with larger ranges don't dominate the model.
- **EDA**: Visualized histograms, correlation heatmaps, and performed PCA (Dimensionality Reduction) to observe the data distribution.
- **Splitting**: Split the dataset into 80% training and 20% testing sets using `train_test_split`.

## Phase 3: Model Selection & Training
- **Models Used**:
    1. **Logistic Regression**: Selected for its efficiency and strong performance on linearly separable binary classification tasks.
    2. **Random Forest Classifier**: Selected to capture non-linear relationships and provide robustness through ensemble learning.
    3. **Voting Classifier (Ensemble)**: Combined both models to leverage their individual strengths.
- **Hyperparameter Tuning**: Used `GridSearchCV` with 5-fold cross-validation to find optimal parameters:
    - Logistic Regression: `C` (regularization strength).
    - Random Forest: `n_estimators` and `max_depth`.

## Phase 4: Model Evaluation & Comparison
- **Metrics**: Accuracy, Precision, Recall, and F1-score were used for evaluation.
- **Cross-Validation**: Performed 5-fold cross-validation on all models to ensure they generalize well and avoid overfitting.
- **Results**:
    - **Logistic Regression**: Achieved high accuracy (~98%) and balanced Precision/Recall.
    - **Random Forest**: Performance was slightly lower but very stable.
    - **Best Model**: Logistic Regression was selected as the best-performing model based on the F1-score on the test set.

## Phase 5: Deployment & Interpretation
- **Deployment**: The best model and scaler were exported using `joblib` and integrated into a **Streamlit** web application (`app.py`).
- **Professional UI/UX Design**: 
    - **Modern Aesthetic**: Used a clean "Medical Dashboard" theme with custom CSS, Google Fonts, and linear gradients.
    - **User-Centric Input**: Diagnostic features are categorized into intuitive sections (Mean, Standard Error, Worst Case) using icon-labeled tabs and slider cards for easy navigation.
    - **Real-time Feedback**: Predictive analysis provides immediate diagnosis with color-coded results and confidence metrics.
- **Real-world Scenario**: This application allows healthcare professionals to input diagnostic measurements and receive an immediate classification, assisting in early cancer detection.
- **Limitations**:
    - Relies on manual feature extraction; deep learning could automate this.
    - Performance may vary on datasets from different clinical settings.
- **Future Improvements**:
    - Adding support for image-based inputs using Convolutional Neural Networks (CNNs).
    - Incorporating more diverse patient data for better generalization.

---
**Prepared by: Junie (AI Agent)**
**Date: 10th April 2026**
