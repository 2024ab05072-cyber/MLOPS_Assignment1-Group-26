import os
import pickle

MODEL_FILE = "artifacts/model.pkl"
SCALER_FILE = "artifacts/scaler.pkl"

def test_model_file_exists():
    """Check if the trained model pickle exists"""
    assert os.path.exists(MODEL_FILE), "Trained model file not found!"

def test_scaler_file_exists():
    """Check if the scaler pickle exists"""
    assert os.path.exists(SCALER_FILE), "Scaler file not found!"

def test_model_loads_correctly():
    """Ensure the model can be deserialized"""
    with open(MODEL_FILE, "rb") as f:
        loaded_model = pickle.load(f)
    assert loaded_model is not None, "Failed to load the model from file"
