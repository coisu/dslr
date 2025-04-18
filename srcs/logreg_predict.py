# # loads weights, makes predictions, outputs houses.csv.
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
    X = X.fillna(0)
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
        class_labels = np.load("trained/class_labels.npy", allow_pickle=True)
        with open("trained/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except Exception as e:
        print(f"Error loading trained model files: {e}")
        sys.exit(1)

    index, X_test = preprocess_data(test_data, scaler)
    predictions = predict(X_test, weights, class_labels)

    unique, counts = np.unique(predictions, return_counts=True)
    print("\n\n>>>House Distribution:")
    for house, count in zip(unique, counts):
        print(f"{house}: {count}")

    df = pd.DataFrame({"Index": index, "Hogwarts House": predictions})

    df.to_csv("outputs/houses.csv", index=False)
    print("Successfully made predictions: outputs/houses.csv saved.")




# import sys
# import numpy as np
# import pandas as pd
# import pickle
# from utils import load_dataset
# from logreg_train import sigmoid

# def preprocess_data(data, scaler):
#     # selected_features = [
#     #     "Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts",
#     #     "Divination", "Muggle Studies", "Ancient Runes", "History of Magic",
#     #     "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"
#     # ]
#     # index = data["Index"].values if "Index" in data.columns else np.arange(len(data))
#     # X = data[selected_features].fillna(0)
#     exclude_columns = {"Index", "First Name", "Last Name", "Birthday", "Best Hand", "Hogwarts House"}
    
#     index = data["Index"].values if "Index" in data.columns else np.arange(len(data))
    
#     y = data["Hogwarts House"]
#     X = data.drop(columns=exclude_columns, errors="ignore").select_dtypes(include=[np.number])
#     X = X.fillna(0)
#     X_scaled = scaler.transform(X)
#     return index, X_scaled

# def predict(X, w, b, class_labels):
#     probs = sigmoid(np.dot(X, w.T) + b.T)  # shape: (n_samples, n_classes)
#     preds = np.argmax(probs, axis=1)
#     return class_labels[preds]

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python logreg_predict.py <test_file.csv>")
#         sys.exit(1)

#     test_file = sys.argv[1]
#     data = load_dataset(test_file, index_col="Index")
#     if data is None:
#         sys.exit(1)

#     try:
#         w = np.load("trained/weights.npy")
#         b = np.load("trained/bias.npy")
#         class_labels = np.load("trained/class_labels.npy", allow_pickle=True)
#         with open("trained/scaler.pkl", "rb") as f:
#             scaler = pickle.load(f)
#     except Exception as e:
#         print("Error loading model files:", e)
#         sys.exit(1)

#     index, X_test = preprocess_data(data, scaler)
#     predictions = predict(X_test, w, b, class_labels)

#     df = pd.DataFrame({"Index": index, "Hogwarts House": predictions})
#     df.to_csv("houses.csv", index=False)

#     print("Predictions saved to houses.csv")
