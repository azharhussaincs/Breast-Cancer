import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import os

# Create a directory for artifacts
os.makedirs('artifacts', exist_ok=True)

# Phase 1: Dataset Selection & Problem Definition
# Dataset: Breast Cancer Wisconsin (Diagnostic)
# Objective: Classify tumors as malignant or benign based on features.
# Type: Supervised Learning (Classification)

def load_and_preprocess():
    print("--- Phase 1 & 2: Loading and Preprocessing ---")
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    print(f"Dataset Shape: {df.shape}")
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    print(f"Missing values: {missing}")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"Duplicate entries: {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates()
        print("Duplicates removed.")

    # EDA: Visualizing key insights
    # 1. Histograms for a few features
    df.iloc[:, :4].hist(figsize=(10, 8), bins=20)
    plt.suptitle("Histograms of First 4 Features")
    plt.savefig('artifacts/histograms.png')
    plt.close()

    # 2. Boxplot for outlier detection
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df.iloc[:, :10])
    plt.xticks(rotation=45)
    plt.title("Boxplot of First 10 Features (Outlier Detection)")
    plt.savefig('artifacts/boxplot.png')
    plt.close()
    
    # 3. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.iloc[:, :10].corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap (First 10 features)")
    plt.savefig('artifacts/correlation_heatmap.png')
    plt.close()
    
    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data (Phase 2: Split dataset into training and testing sets)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling (Phase 2: Scaling/normalization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, 'artifacts/scaler.joblib')
    
    return X_train_scaled, X_test_scaled, y_train, y_test, data.feature_names

# Phase 3: Model Selection & Training
def train_models(X_train, y_train):
    print("\n--- Phase 3: Model Selection & Training ---")
    
    # Model 1: Logistic Regression
    lr = LogisticRegression(max_iter=10000)
    lr_params = {'C': [0.1, 1, 10]}
    gs_lr = GridSearchCV(lr, lr_params, cv=5)
    gs_lr.fit(X_train, y_train)
    print(f"Best Logistic Regression Params: {gs_lr.best_params_}")
    
    # Model 2: Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf_params = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
    gs_rf = GridSearchCV(rf, rf_params, cv=5)
    gs_rf.fit(X_train, y_train)
    print(f"Best Random Forest Params: {gs_rf.best_params_}")
    
    # Bonus: Ensemble Model (Voting)
    ensemble = VotingClassifier(estimators=[
        ('lr', gs_lr.best_estimator_),
        ('rf', gs_rf.best_estimator_)
    ], voting='soft')
    ensemble.fit(X_train, y_train)
    
    return gs_lr.best_estimator_, gs_rf.best_estimator_, ensemble

# Phase 4: Model Evaluation
def evaluate_models(models, X_train, y_train, X_test, y_test):
    print("\n--- Phase 4: Model Evaluation & Comparison ---")
    results = {}
    for name, model in models.items():
        # Metrics on test set
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Cross-validation (Phase 4: Perform cross-validation to ensure generalization)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        cv_mean = cv_scores.mean()
        
        results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'CV_Mean': cv_mean}
        
        print(f"\nModel: {name}")
        print(f"Test Accuracy: {acc:.4f}, CV Mean Accuracy: {cv_mean:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix: {name}")
        plt.savefig(f'artifacts/cm_{name.replace(" ", "_")}.png')
        plt.close()

    # Select best model (Ensemble in this case usually)
    best_model_name = max(results, key=lambda k: results[k]['F1'])
    print(f"\nBest performing model based on F1-score: {best_model_name}")
    
    # Save best model
    joblib.dump(models[best_model_name], 'artifacts/best_model.joblib')
    return best_model_name

# Bonus: PCA
def perform_pca(X_train, X_test):
    print("\n--- Bonus: Dimensionality Reduction (PCA) ---")
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c='blue', alpha=0.5)
    plt.title("PCA of Breast Cancer Dataset (First 2 components)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig('artifacts/pca_plot.png')
    plt.close()
    print("PCA plot saved.")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features = load_and_preprocess()
    perform_pca(X_train, X_test)
    lr_model, rf_model, ensemble_model = train_models(X_train, y_train)
    
    models_to_eval = {
        'Logistic Regression': lr_model,
        'Random Forest': rf_model,
        'Ensemble (Voting)': ensemble_model
    }
    
    evaluate_models(models_to_eval, X_train, y_train, X_test, y_test)
    print("\nTraining and Evaluation complete. Artifacts saved in 'artifacts/'.")
