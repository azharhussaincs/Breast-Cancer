# Machine Learning Project Report: Breast Cancer Classification

## Phase 1: Dataset Selection & Problem Definition
- **Dataset**: Breast Cancer Wisconsin (Diagnostic) Dataset.
- **Introduction**: The dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.
- **Objective**: To develop a classification model that can predict whether a tumor is **Malignant** (0) or **Benign** (1) based on these features.
- **ML Type**: Supervised Learning (Binary Classification).

## Phase 2: Data Preprocessing & Exploratory Data Analysis (EDA)
- **Data Loading**: Loaded using `sklearn.datasets.load_breast_cancer`.
- **Handling Missing Values**: The dataset had no missing values.
- **Feature Engineering**: Features were scaled using `StandardScaler` to normalize the data.
- **EDA**: Visualized correlation heatmaps and performed PCA (Dimensionality Reduction) to observe the data distribution in lower dimensions.
- **Splitting**: Split the dataset into 80% training and 20% testing sets.

## Phase 3: Model Selection & Training
- **Models Used**:
    1. **Logistic Regression**: A linear model suitable for binary classification tasks.
    2. **Random Forest Classifier**: An ensemble learning method based on decision trees.
    3. **Voting Classifier (Ensemble)**: Combined Logistic Regression and Random Forest to improve robustness.
- **Hyperparameter Tuning**: Performed Grid Search with Cross-Validation (5-fold) for both Logistic Regression (C parameter) and Random Forest (n_estimators, max_depth).

## Phase 4: Model Evaluation & Comparison
- **Metrics**: Accuracy, Precision, Recall, and F1-score.
- **Results**:
    - **Logistic Regression**: F1-score ~ 0.98 (Best performing model in this case).
    - **Random Forest**: F1-score ~ 0.97.
    - **Ensemble Model**: F1-score ~ 0.97.
- **Cross-Validation**: Performed to ensure the models generalize well to unseen data.

## Phase 5: Deployment & Interpretation
- **Deployment**: The best-performing model (Logistic Regression) was converted into a deployable format using `joblib` and integrated into a **Streamlit** web application.
- **Real-world Scenario**: This model can serve as a diagnostic tool for radiologists to assist in identifying cancerous tumors from FNA features, providing a "second opinion" or rapid screening.
- **Limitations**:
    - The model depends on the quality of feature extraction from images.
    - It should not replace a doctor's final diagnosis but act as an auxiliary tool.
- **Future Improvements**:
    - Integration of Deep Learning models (CNNs) directly on digitized images.
    - Incorporation of more diverse patient demographics.

---
**Prepared by: Junie (AI Agent)**
**Date: 10th April 2026**
