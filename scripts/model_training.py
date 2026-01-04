# Required Imports
import os
import pandas as pd
import numpy as np
import pickle
import warnings

# ML utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Experiment tracking
import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri("mlruns")
# Ignore unnecessary warnings
warnings.filterwarnings("ignore")

# Path Configuration
DATA_FILE = "dataset/heart.csv"
OUTPUT_DIR = "artifacts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize MLflow experiment
mlflow.set_experiment("Heart Disease Prediction")

# Load & Prepare Dataset
df = pd.read_csv(DATA_FILE)

# Replace placeholder values and impute missing data
df['ca'] = df['ca'].replace('?', np.nan)
df['thal'] = df['thal'].replace('?', np.nan)

df['ca'].fillna(df['ca'].mode()[0], inplace=True)
df['thal'].fillna(df['thal'].mode()[0], inplace=True)

# Separate input features and target
features = df.drop('num', axis=1)
target = df['num'].apply(lambda val: 1 if val > 0 else 0)

# Train–test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.30,
    random_state=4,
    stratify=target
)

# Feature Normalization
scaler_obj = StandardScaler()
X_train = scaler_obj.fit_transform(X_train)
X_test = scaler_obj.transform(X_test)

# Persist scaler for reuse
with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler_obj, f)

# MLflow Logging Function
def run_and_log(model_instance, run_label, Xtr, ytr, Xte, yte, scaled=False, scaler=None):
    with mlflow.start_run(run_name=run_label):

        # Log model metadata
        mlflow.log_param("model_type", run_label)
        for param, val in model_instance.get_params().items():
            mlflow.log_param(param, val)

        # Train model
        model_instance.fit(Xtr, ytr)

        # Generate predictions
        preds = model_instance.predict(Xte)
        probs = model_instance.predict_proba(Xte)[:, 1]

        # Compute evaluation metrics
        acc = accuracy_score(yte, preds)
        prec = precision_score(yte, preds)
        rec = recall_score(yte, preds)
        f1 = f1_score(yte, preds)
        auc = roc_auc_score(yte, probs)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        # Store trained model
        mlflow.sklearn.log_model(model_instance, "model")

        print(f"Experiment logged for: {run_label}")

        return auc


# Model Experiments

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
lr_auc = run_and_log(
    log_reg,
    "Logistic Regression",
    X_train, y_train,
    X_test, y_test,
    scaled=True,
    scaler=scaler_obj
)

# Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_auc = run_and_log(
    rf_clf,
    "Random Forest",
    X_train, y_train,
    X_test, y_test
)

# AdaBoost
ada_clf = AdaBoostClassifier(n_estimators=250, random_state=42)
ada_auc = run_and_log(
    ada_clf,
    "AdaBoost",
    X_train, y_train,
    X_test, y_test
)


# Best Model Selection
auc_results = {
    "Logistic Regression": lr_auc,
    "Random Forest": rf_auc,
    "AdaBoost": ada_auc
}

best_model_key = max(auc_results, key=auc_results.get)
print(f"Best performing model: {best_model_key} | ROC-AUC: {auc_results[best_model_key]}")

best_model = {
    "Logistic Regression": log_reg,
    "Random Forest": rf_clf,
    "AdaBoost": ada_clf
}[best_model_key]

# Save best model artifact
with open(os.path.join(OUTPUT_DIR, "model.pkl"), "wb") as f:
    pickle.dump(best_model, f)
