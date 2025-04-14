# loads weights, makes predictions, outputs houses.csv.
from utils import load_dataset
from logreg_train import sigmoid
import sys
import numpy as np
import pandas as pd
import pickle
from typing import Tuple


def predict(X: np.ndarray, theta: np.ndarray, class_labels: np.ndarray) -> np.ndarray:
    probabilities = sigmoid(X @ theta.T)
    index = np.argmax(probabilities, axis=1)
    return class_labels[index]

def preprocess_data(data: pd.DataFrame, scaler) -> Tuple[np.ndarray, np.ndarray]:
    exclude_columns = {"Index", "First Name", "Last Name", "Birthday", "Best Hand", "Hogwarts House"}
    index = data["Index"].values if "Index" in data.columns else np.arange(len(data))
    
    y = data["Hogwarts House"]
    X = data.drop(columns=exclude_columns, errors="ignore").select_dtypes(include=[np.number])
    
    X_scaled = scaler.transform(X)
    X_with_bias = np.c_[np.ones(X_scaled.shape[0]), X_scaled]
    
    return index, X_with_bias

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python logreg_predict.py <file_path>")
        sys.exit(1)
    file_path = sys.argv[1]
    test_data = load_dataset(file_path, index_col=None)
    if test_data is None:
        sys.exit(1)

    try:
        weights = np.load("trained/weights.npy")
        class_labels = np.load("trained/class_labels.npy")
        with open("trained/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except Exception as e:
        print(f"Error loading trained model files: {e}")
        sys.exit(1)

    index, X_test = preprocess_data(test_data, scaler)
    predictions = predict(X_test, weights, class_labels)

    df = pd.DataFrame({"Index": index, "Hogwarts House": predictions})

    df.to_csv("houses.csv", index=False)
    print("✅ Successfully made predictions: houses.csv saved.")