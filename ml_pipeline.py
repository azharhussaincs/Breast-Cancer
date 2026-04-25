# Bringing in the essential libraries for our machine learning journey
import pandas as pd  # For data manipulation and handling dataframes
import numpy as np   # For numerical operations and array handling
import matplotlib.pyplot as plt  # For creating static visualizations
import seaborn as sns  # For making beautiful statistical plots
from sklearn.datasets import load_breast_cancer  # To load the built-in breast cancer dataset
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score  # Tools for splitting data and tuning models
from sklearn.preprocessing import StandardScaler  # To scale our features so they're on the same level
from sklearn.decomposition import PCA  # For dimensionality reduction (making complex data simpler to see)
from sklearn.linear_model import LogisticRegression  # Our first model choice: reliable and simple
from sklearn.ensemble import RandomForestClassifier, VotingClassifier  # Powerful ensemble models for better accuracy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report  # To see how well we're doing
import joblib  # To save our trained models so we can use them later in the app
import os      # For interacting with the operating system (like creating folders)

# First things first, let's make sure we have a place to save our results
# This creates an 'artifacts' folder if it doesn't already exist
os.makedirs('artifacts', exist_ok=True)

# Phase 1 & 2: Getting the data ready
# Here we'll load the dataset, clean it up a bit, and do some basic exploration
def load_and_preprocess():
    print("--- Phase 1 & 2: Loading and Preprocessing ---")
    
    # Load the actual data from scikit-learn
    data = load_breast_cancer()
    
    # Put the features into a nice table format (DataFrame)
    df = pd.DataFrame(data.data, columns=data.feature_names)
    
    # Add the 'target' column which tells us if it's Malignant or Benign
    df['target'] = data.target
    
    # Just a quick check to see how much data we're working with
    print(f"Dataset Shape: {df.shape}")
    
    # We need to make sure there are no holes in our data (missing values)
    missing = df.isnull().sum().sum()
    print(f"Missing values: {missing}")
    
    # And check if we accidentally have the same patient records twice
    duplicates = df.duplicated().sum()
    print(f"Duplicate entries: {duplicates}")
    if duplicates > 0:
        # If we find any, we toss them out to keep the data clean
        df = df.drop_duplicates()
        print("Duplicates removed.")

    # Time for some Visual Exploration (EDA)
    # 1. Let's look at the distribution of the first 4 features using histograms
    df.iloc[:, :4].hist(figsize=(10, 8), bins=20)
    plt.suptitle("Histograms of First 4 Features")
    plt.savefig('artifacts/histograms.png') # Save the plot for our report
    plt.close() # Close it to save memory

    # 2. Use boxplots to spot any weird values (outliers) in the first 10 features
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df.iloc[:, :10])
    plt.xticks(rotation=45) # Tilt the labels so they don't overlap
    plt.title("Boxplot of First 10 Features (Outlier Detection)")
    plt.savefig('artifacts/boxplot.png')
    plt.close()
    
    # 3. Create a heatmap to see which features are linked to each other
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.iloc[:, :10].corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap (First 10 features)")
    plt.savefig('artifacts/correlation_heatmap.png')
    plt.close()
    
    # Now, let's separate the features (X) from what we're trying to predict (y)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # We'll split the data: 80% for training the brain, 20% for testing how smart it got
    # Using a random_state of 42 so we get the same results every time we run this
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling is crucial! We want all our numbers to be in a similar range
    # This helps models like Logistic Regression work much better
    scaler = StandardScaler()
    
    # We 'fit' the scaler on the training data and then transform both sets
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler artifact so the app can use the exact same scaling logic later
    joblib.dump(scaler, 'artifacts/scaler.joblib')
    
    # Send back the processed data so we can start training
    return X_train_scaled, X_test_scaled, y_train, y_test, data.feature_names

# Phase 3: Teaching the models
# We'll try a few different approaches and pick the best one
def train_models(X_train, y_train):
    print("\n--- Phase 3: Model Selection & Training ---")
    
    # Model 1: Logistic Regression - Simple, fast, and often very effective for this kind of data
    lr = LogisticRegression(max_iter=10000)
    # We'll test different levels of 'strictness' (C) to find the sweet spot
    lr_params = {'C': [0.1, 1, 10]}
    # GridSearchCV tries all combinations and picks the best one using 5-fold cross-validation
    gs_lr = GridSearchCV(lr, lr_params, cv=5)
    gs_lr.fit(X_train, y_train)
    print(f"Best Logistic Regression Params: {gs_lr.best_params_}")
    
    # Model 2: Random Forest - A more complex model that uses many 'decision trees'
    rf = RandomForestClassifier(random_state=42)
    # We'll tune how many trees (n_estimators) and how deep they go (max_depth)
    rf_params = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
    gs_rf = GridSearchCV(rf, rf_params, cv=5)
    gs_rf.fit(X_train, y_train)
    print(f"Best Random Forest Params: {gs_rf.best_params_}")
    
    # Bonus: Let's combine them into an Ensemble! 
    # This 'Voting Classifier' takes the average of both models for a more robust prediction
    ensemble = VotingClassifier(estimators=[
        ('lr', gs_lr.best_estimator_),
        ('rf', gs_rf.best_estimator_)
    ], voting='soft')
    ensemble.fit(X_train, y_train)
    
    # Return the best versions of each model
    return gs_lr.best_estimator_, gs_rf.best_estimator_, ensemble

# Phase 4: Seeing how well they actually work
def evaluate_models(models, X_train, y_train, X_test, y_test):
    print("\n--- Phase 4: Model Evaluation & Comparison ---")
    results = {}
    
    # Loop through each model and check its performance
    for name, model in models.items():
        # Get predictions on the test data that the model has never seen before
        y_pred = model.predict(X_test)
        
        # Calculate various scores to get the full picture
        acc = accuracy_score(y_test, y_pred) # Overall correctness
        prec = precision_score(y_test, y_pred) # How many of our 'positive' guesses were right?
        rec = recall_score(y_test, y_pred) # How many of the actual 'positives' did we catch?
        f1 = f1_score(y_test, y_pred) # A balance between precision and recall
        
        # Cross-validation helps us be sure the model didn't just 'get lucky' on one split
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        cv_mean = cv_scores.mean()
        
        # Store the results for comparison
        results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'CV_Mean': cv_mean}
        
        # Print out a detailed report for this model
        print(f"\nModel: {name}")
        print(f"Test Accuracy: {acc:.4f}, CV Mean Accuracy: {cv_mean:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Let's visualize the mistakes with a Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix: {name}")
        # Save the matrix image so we can show it in the app
        plt.savefig(f'artifacts/cm_{name.replace(" ", "_")}.png')
        plt.close()

    # We'll pick our "Best Model" based on the F1-score
    best_model_name = max(results, key=lambda k: results[k]['F1'])
    print(f"\nBest performing model based on F1-score: {best_model_name}")
    
    # Save this champion model as a file for our Streamlit app to use
    joblib.dump(models[best_model_name], 'artifacts/best_model.joblib')
    return best_model_name

# Bonus: Reducing dimensions to 2D so we can actually plot the data
def perform_pca(X_train, X_test):
    print("\n--- Bonus: Dimensionality Reduction (PCA) ---")
    
    # PCA squashes the 30 features down into just 2 main components
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    
    # Plot the result to see if the classes are naturally separated
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c='blue', alpha=0.5)
    plt.title("PCA of Breast Cancer Dataset (First 2 components)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig('artifacts/pca_plot.png')
    plt.close()
    print("PCA plot saved.")

# This is the main engine room that runs everything in order
if __name__ == "__main__":
    # 1. Load and clean
    X_train, X_test, y_train, y_test, features = load_and_preprocess()
    
    # 2. Visualize the data space
    perform_pca(X_train, X_test)
    
    # 3. Train all our model candidates
    lr_model, rf_model, ensemble_model = train_models(X_train, y_train)
    
    # 4. Put them in a dictionary for easy evaluation
    models_to_eval = {
        'Logistic Regression': lr_model,
        'Random Forest': rf_model,
        'Ensemble (Voting)': ensemble_model
    }
    
    # 5. Evaluate and pick the winner
    evaluate_models(models_to_eval, X_train, y_train, X_test, y_test)
    
    print("\nTraining and Evaluation complete. All artifacts are safe in the 'artifacts/' folder.")
