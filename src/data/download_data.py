import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

OUTPUT_DIR = "data"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "heart.csv")

def download_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Fetch Heart Disease dataset from UCI
    heart_disease = fetch_ucirepo(id=45)

    # Features and target
    X = heart_disease.data.features
    y = heart_disease.data.targets

    # Combine features and target
    df = pd.concat([X, y], axis=1)

    # Save to CSV
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Dataset downloaded from UCI and saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    download_dataset()
