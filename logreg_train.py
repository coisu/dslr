# trains one-vs-all logistic regression using gradient descent, saves weights
# must not use libraries like scikit-learn or built-in logistic regression implementations.
# build it from scratch using NumPy.

# 1. Load training dataset.
# 2. Extract feature matrix X and label vector y.
# 3. Train four binary classifiers (one per house) using logistic regression via gradient descent.
# 4. Save resulting weight vectors into a file (e.g., weights.csv or .npy).


import numpy as np
import pandas as pd
import sys
from utils import load_dataset
from sklearn.preprocessing import StandardScaler

from typing import Tuple, List



def preprocess_data(data: pd.DataFrame, scaler) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    exclude_columns = {"Index", "First Name", "Last Name", "Birthday", "Best Hand", "Hogwarts House"}

    X = data.drop(columns=exclude_columns, errors="ignore").select_dtypes(include=[np.number])
    y = data["Hogwarts House"]

    X_scaled = scaler.fit_transform(X)

    X_with_bias = np.c_[np.ones(X_scaled.shape[0]), X_scaled]

    return X_with_bias, y.to_numpy(), np.unique(y)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pair_plot.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    data = load_dataset(file_path, index_col="Index")
    if data is None:
        sys.exit(1)
    
    scaler = StandardScaler()

    X, y, class_labels = preprocess_data(data)
    theta_matrix = train_one_vs_all()

    np.save("weights.npy", theta_matrix)
    np.save("class_labels.npy", class_labels)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("✅ Successfully model trained: weights.npy, class_labels.npy, scaler.pkl saved")