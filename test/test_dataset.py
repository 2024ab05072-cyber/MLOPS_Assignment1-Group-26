import os
import pandas as pd

DATA_FILE = "data/heart.csv"

def test_dataset_file_presence():
    """Check if the dataset CSV exists"""
    assert os.path.exists(DATA_FILE), "Heart dataset file is missing!"

def test_dataset_loads_successfully():
    """Check if dataset loads without being empty"""
    df = pd.read_csv(DATA_FILE)
    assert not df.empty, "Loaded dataset is empty!"

def test_target_column_present():
    """Ensure the target column 'num' exists in the dataset"""
    df = pd.read_csv(DATA_FILE)
    assert "num" in df.columns, "Target column 'num' not found in dataset"
