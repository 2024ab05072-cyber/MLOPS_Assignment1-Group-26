# Import necessary libraries

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=UserWarning)

# For Model training and evaluation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
from lightgbm import LGBMClassifier   

# For Experiment tracking
import mlflow
import mlflow.sklearn

# Suppressing warnings
import warnings
warnings.filterwarnings("ignore")

# Defining Data Path & Artifact Path
DATA_PATH = "data/heart.csv"
ARTIFACT_PATH = "artifacts"

os.makedirs(ARTIFACT_PATH, exist_ok=True)

# Setting up MLflow experiment
mlflow.set_experiment("Heart Disease Prediction")

# Loading the dataset
data = pd.read_csv(DATA_PATH)

# Handling missing values
data['ca'] = data['ca'].replace('?', np.nan)
data['thal'] = data['thal'].replace('?', np.nan)
data['ca'].fillna(data['ca'].mode()[0], inplace=True)
data['thal'].fillna(data['thal'].mode()[0], inplace=True)

# Splitting features and target variable
X = data.drop('num', axis=1)
y = data['num'].apply(lambda x: 1 if x > 0 else 0)  # Binary classification

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=4, stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Saving the scaler
with open(os.path.join(ARTIFACT_PATH, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# Function to run experiments with MLflow
def log_experiment(model, model_name, X_train, y_train, X_test, y_test, scaled=False, scaler=None):
    with mlflow.start_run(run_name=model_name):
        # Log params
        mlflow.log_param("model", model_name)
        for key, value in model.get_params().items():
            mlflow.log_param(key, value)

        # Train the model
        if scaled and scaler is not None:
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        else:
            X_train = X_train
            X_test = X_test

        model.fit(X_train, y_train)

        # Predictions and probabilities
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]

        # Evaluation Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_probs)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Log the model
        mlflow.sklearn.log_model(model, "model")

        print(f"Logged {model_name} to MLflow")

        return roc_auc_score(y_test, y_probs)

# Model Training
# Model 1: Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)

lr_auc = log_experiment(lr_model, "Logistic Regression", X_train, y_train, X_test, y_test,
                        scaled=True, scaler=scaler)

# Model 2: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_auc = log_experiment(rf_model, "Random Forest", X_train, y_train, X_test, y_test)

# Model 3: LightGBM (replacing XGBoost)
lgbm_model = LGBMClassifier(n_estimators=300, random_state=42, verbose=-1)

lgbm_auc = log_experiment(lgbm_model, "LightGBM", X_train, y_train, X_test, y_test)

# Selecting the best model based on ROC-AUC
model_aucs = {
    "Logistic Regression": lr_auc,
    "Random Forest": rf_auc,
    "LightGBM": lgbm_auc
}

best_model_name = max(model_aucs, key=model_aucs.get)
print(f"Best Model: {best_model_name} with ROC-AUC: {model_aucs[best_model_name]}")

# Retrieving the best model instance
BEST_MODEL = {
    "Logistic Regression": lr_model,
    "Random Forest": rf_model,
    "LightGBM": lgbm_model
}[best_model_name]

# Saving the best model
with open(os.path.join(ARTIFACT_PATH, "model.pkl"), "wb") as f:
    pickle.dump(BEST_MODEL, f)
