import numpy as np
from utils import load_dataset
from sklearn.preprocessing import StandardScaler
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py <dataset_train.csv>")
        return

    # Load the training dataset
    dataset_path = sys.argv[1]
    dataset_path = sys.argv[1]
    data = load_dataset(dataset_path)
    if data is None:
        return

    # Extract features (X) and labels (y)
    features = data.drop(columns=['Hogwarts House']).values
    labels = data['Hogwarts House'].map({
        'Gryffindor': 0,
        'Hufflepuff': 1,
        'Ravenclaw': 2,
        'Slytherin': 3
    }).values

    # Standardize features
    # X = (features - features.mean(axis=0)) / features.std(axis=0): same as StandardScaler
    # (features - features.mean(axis=0)) set the mean of each feature to 0
    # / features.std(axis=0) set the standard deviation of each feature to 1
    # X is now standardized 2D array of features
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    X = np.c_[np.ones(X.shape[0]), X]
    # X.shape[0] is the number of rows in X
    # shape returns a tuple (rows, columns) in NumPy
    # (rows, columns): e.g. (1600, 13) -> 1600 students, 13 features
    # X.shape[0]: returns the number of rows (1600 students)
    # X.shape[1]: returns the number of columns (13 features)

    # np.ones(n) creates an array of n ones
    # np.ones(5) -> [1, 1, 1, 1, 1]

    # np.c_: combines two arrays by columns
    # np.c_[np.array([1, 2, 3]), np.array([4, 5, 6])] -> [[1, 4], [2, 5], [3, 6]]
    # X = np.array([[10, 20], [30, 40], [50, 60]])
    # np.c_[np.ones(X.shape[0]), X] -> [[1, 10, 20], [1, 30, 40], [1, 50, 60]]

    # this process adds a intercept term to the features
    # z= θ0 + θ1x1 + θ2x2 +⋯+ θnxn : fomula for linear regression(logistic regression)
    # θ0 is the intercept term
    # θ1, θ2, ..., θn are the weights for the features (weights vector)
    # x1, x2, ..., xn are the features (data * size)
    # when evey feature is 0, we still get valid value(the intercept term is the output)
    # to include the intercept term in the weights vector, we add a column of ones(1) to the features
    # then now we have n+1 features (θ0*1)


    y = labels

    # Train the model
    num_classes = 4  # Gryffindor, Hufflepuff, Ravenclaw, Slytherin
    learning_rate = 0.01
    iterations = 1000
    weights = train_one_vs_all(X, y, num_classes, learning_rate, iterations)

    # Save the weights
    np.save("weights.npy", weights)
    print("Training completed. Weights saved to 'weights.npy'.")

if __name__ == "__main__":
    main()