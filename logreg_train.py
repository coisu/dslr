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
    z = np.clip(z, -500, 500)  # Prevent overflow
    return 1 / (1 + np.exp(-z))

def gradient_descent(X: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float, num_iters: int, lambda_:float = 0.1) -> np.ndarray:
    m = len(y)
    for _ in range(num_iters):
        predictions = sigmoid(X @ theta)
        # gradient = (1 / m) * (X.T @ (predictions - y)) + (lambda_ / m) * theta
        gradient = (1 / m) * (X.T @ (predictions - y))
        
        # L2 regularization (excluding bias term)
        reg_term = (lambda_ / m) * theta
        reg_term[0] = 0  # Don't regularize the bias term
        gradient += reg_term

        theta -= alpha * gradient
    return theta

def train_one_vs_all(X: np.ndarray, y: np.ndarray, labels: np.ndarray, alpha: float = 0.05, num_iters: int = 5000, lambda_: float = 0.1) -> np.ndarray:
    m, n = X.shape
    theta_matrix = np.zeros((len(labels), n))

    for i, label in enumerate(labels):
        y_binary = (y == label).astype(int)
        theta = np.zeros(n)
        theta = gradient_descent(X, y_binary, theta, alpha, num_iters, lambda_)
        theta_matrix[i] = theta
    print("theta_matrix:\n", theta_matrix)
    return theta_matrix

def preprocess_data(data: pd.DataFrame, scaler) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    exclude_columns = {"Index", "First Name", "Last Name", "Birthday", "Best Hand", "Hogwarts House"}

    y = data["Hogwarts House"]
    X = data.drop(columns=exclude_columns, errors="ignore").select_dtypes(include=[np.number])
    X = X.fillna(0)
    y = y[X.index]

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

    X, y, class_labels = preprocess_data(data, scaler)
    print("\n\n>>>\n")
    print(np.unique(y, return_counts=True))

    print("\n\nDebugging info:")
    print("Any NaN in X:", np.isnan(X).any())
    print("Any Inf in X:", np.isinf(X).any())

    theta_matrix = train_one_vs_all(X, y, class_labels, alpha=0.05, num_iters=5000, lambda_=0.1)

    os.makedirs("trained", exist_ok=True)

    np.save("trained/weights.npy", theta_matrix)        # Save weights
    np.save("trained/class_labels.npy", class_labels)   # Save class labels (Gryffindor, Ravenclaw, Hufflepuff, Slytherin)
    with open("trained/scaler.pkl", "wb") as f:         # standardize the data
        pickle.dump(scaler, f)

    print("✅ Successfully model trained[implicit bias]: weights.npy, class_labels.npy, scaler.pkl saved in trained/")







# import numpy as np
# import pandas as pd
# import sys
# import os
# import pickle
# from utils import load_dataset
# from sklearn.preprocessing import StandardScaler
# from typing import Tuple

# def sigmoid(z: np.ndarray) -> np.ndarray:
#     z = np.clip(z, -500, 500)
#     return 1 / (1 + np.exp(-z))

# def gradient_descent(X, y, w, b, alpha, num_iters, lambda_=0.1):
#     m = X.shape[0]
#     for _ in range(num_iters):
#         preds = sigmoid(np.dot(X, w.T) + b)
#         error = preds - y.reshape(-1, 1)

#         dw = (1 / m) * np.dot(error.T, X) + (lambda_ / m) * w
#         db = (1 / m) * np.sum(error)

#         w -= alpha * dw
#         b -= alpha * db
#     return w, b

# def train_one_vs_all(X, y, labels, alpha=0.05, num_iters=5000, lambda_=0.1):
#     w_all = np.zeros((len(labels), X.shape[1]))
#     b_all = np.zeros((len(labels), 1))
#     for i, label in enumerate(labels):
#         y_binary = (y == label).astype(int)
#         w = np.zeros((1, X.shape[1]))
#         b = 0
#         w, b = gradient_descent(X, y_binary, w, b, alpha, num_iters, lambda_)
#         w_all[i] = w
#         b_all[i] = b
#     return w_all, b_all

# def preprocess_data(data: pd.DataFrame, scaler) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     exclude_columns = {"Index", "First Name", "Last Name", "Birthday", "Best Hand", "Hogwarts House"}
#     y = data["Hogwarts House"]
#     X = data.drop(columns=exclude_columns, errors="ignore").select_dtypes(include=[np.number])
#     X = X.fillna(0)
#     y = y[X.index]
#     X_scaled = scaler.fit_transform(X)
#     return X_scaled, y.to_numpy(), np.unique(y)

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python logreg_train.py <file_path>")
#         sys.exit(1)

#     file_path = sys.argv[1]
#     data = load_dataset(file_path, index_col="Index")
#     if data is None:
#         sys.exit(1)

#     scaler = StandardScaler()
#     X, y, class_labels = preprocess_data(data, scaler)

#     w, b = train_one_vs_all(X, y, class_labels, alpha=0.05, num_iters=5000, lambda_=0.1)

#     os.makedirs("trained", exist_ok=True)
#     np.save("trained/weights.npy", w)
#     np.save("trained/bias.npy", b)
#     np.save("trained/class_labels.npy", class_labels)
#     with open("trained/scaler.pkl", "wb") as f:
#         pickle.dump(scaler, f)

    # print("✅ Successfully model trained[explicit bias]: weights.npy, class_labels.npy, scaler.pkl saved in trained/")
