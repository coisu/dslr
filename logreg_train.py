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
import os
import pickle

from utils import load_dataset
from sklearn.preprocessing import StandardScaler

from typing import Tuple, List


# sigmoid Function
# What matters isn't just a binary yes or no 
# we want to know how likely it is that a student belongs to Gryffindor.
# The sigmoid function transforms the model’s raw output into a probability between 0 and 1, 
# which we then use to make our classification decision.

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

def gradient_descent(X: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float, num_iters: int) -> np.ndarray:
    m = len(y)
    for _ in range(num_iters):
        predictions = sigmoid(X @ theta)
        gradient = (1 / m) * (X.T @ (predictions - y))
        theta -= alpha * gradient
    return theta

def train_one_vs_all(X: np.ndarray, y: np.ndarray, labels: np.ndarray, alpha: float = 0.01, num_iters: int = 1000) -> np.ndarray:
    m, n = X.shape
    theta_matrix = np.zeros(len(labels, n))

    for i, label in enumerate(labels):
        y_binary = (y == label).astype(int)
        theta = np.zeros(n)
        theta = gradient_descent(X, y_binary, theta, alpha, num_iters)
        theta_matrix[i] = theta
    return theta_matrix

def preprocess_data(data: pd.DataFrame, scaler) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    exclude_columns = {"Index", "First Name", "Last Name", "Birthday", "Best Hand", "Hogwarts House"}

    y = data["Hogwarts House"]
    X = data.drop(columns=exclude_columns, errors="ignore").select_dtypes(include=[np.number])

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

    os.makedirs("trained", exist_ok=True)

    np.save("trained/weights.npy", theta_matrix)        # Save weights
    np.save("trained/class_labels.npy", class_labels)   # Save class labels (Gryffindor, Ravenclaw, Hufflepuff, Slytherin)
    with open("trained/scaler.pkl", "wb") as f:         # standardize the data
        pickle.dump(scaler, f)

    print("✅ Successfully model trained: weights.npy, class_labels.npy, scaler.pkl saved in trained/")