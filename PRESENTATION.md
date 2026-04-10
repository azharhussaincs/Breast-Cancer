# Presentation: Machine Learning Project - Breast Cancer Classification

## 1. Introduction & Problem Definition
- **Dataset**: Breast Cancer Wisconsin (Diagnostic) Dataset.
- **Problem**: Binary classification to distinguish between Malignant and Benign tumors.
- **Goal**: Build a reliable ML pipeline to assist in diagnosis based on cell nuclei features.
- **ML Type**: Supervised Learning.

## 2. Data Preprocessing & EDA
- **Data Quality**: Checked for missing values and duplicates (none found).
- **Outlier Detection**: Boxplots used to identify potential outliers in feature distributions.
- **Normalization**: Applied `StandardScaler` to ensure all features contribute equally.
- **Insights**: Correlation analysis revealed highly redundant features (e.g., radius, perimeter, and area).
- **Visualization**: Histograms and Heatmaps used to understand feature distribution and relationships.

## 3. Dimensionality Reduction (Bonus)
- **Method**: Principal Component Analysis (PCA).
- **Outcome**: Reduced 30 features to 2 principal components.
- **Result**: Significant separation between classes observed in the 2D PCA space.

## 4. Model Selection & Training
- **Models Used**: 
    1. Logistic Regression (Linear baseline).
    2. Random Forest (Non-linear ensemble).
    3. Voting Classifier (Bonus: Ensemble of LR and RF).
- **Optimization**: Used `GridSearchCV` for hyperparameter tuning.
- **Validation**: 5-fold cross-validation implemented to ensure generalization.

## 5. Model Evaluation & Comparison
- **Metrics**: Accuracy, Precision, Recall, and F1-score.
- **Performance**:
    - **Logistic Regression**: F1 ~ 0.98.
    - **Random Forest**: F1 ~ 0.97.
- **Comparison**: Logistic Regression performed best due to the linear separability of the dataset features.

## 6. Deployment & Interpretation
- **Web App**: Built using **Streamlit**.
- **Features**: Real-time prediction sidebar with adjustable sliders for diagnostic features.
- **Use Case**: Quick screening tool for healthcare providers.
- **Limitations**: Requires expert-curated input data; not a replacement for medical diagnosis.

## 7. Conclusion & Future Work
- **Summary**: Successfully implemented a high-accuracy classification pipeline.
- **Future**: 
    - Integrate CNNs for direct image analysis.
    - Expand dataset to include more clinical variability.

---
**Prepared by: Junie**
**Date: 10th April 2026**
