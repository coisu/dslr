import sys
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import load_dataset

# feature1 and feature2 are the names of the features to plot(x and y axis)
def plot_scatter(data, feature1, feature2, output_path="scatter_plot.png"):
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    house_colors = {
        "Gryffindor": "red",
        "Hufflepuff": "orange",
        "Ravenclaw": "blue",
        "Slytherin": "green"
    }

    plt.figure(figsize=(8, 6))  # Create matplotlib figure object for graph. Set the size of the plot(8x6 inches)
    for house in data['Hogwarts House'].unique():
        subset = data[data['Hogwarts House'] == house]
        plt.scatter(subset[feature1], subset[feature2], label=house, alpha=0.5, s=10, color=house_colors.get(house, "gray"))

    plt.title(f"Scatter Plot: {feature1} vs {feature2}") # Set the title of the plot
    plt.xlabel(feature1) # Set the label (name) of the x-axis
    plt.ylabel(feature2) # Set the label (name) of the y-axis
    plt.legend(title="Hogwarts House")  # Add a legend to the plot
    plt.grid(True) # Add a grid to the plot
    # plt.show()
    plt.savefig(output_path, format='png', dpi=300)
    print(f"Scatter plot saved to {output_path}")

# data = pandas DataFrame
def find_most_similar_features(data, scale_data=False):

    exclude_columns = {"Index", "First Name", "Last Name", "Birthday", "Best Hand", "Hogwarts House"}
    numeric_data = data.drop(columns=exclude_columns, errors='ignore').select_dtypes(include=[np.number])

    # check the Readme.md on github, 'Feature Scaling' section
    if scale_data:
        scaler = StandardScaler()   # Create a StandardScaler object
        numeric_data = pd.DataFrame(scaler.fit_transform(numeric_data), columns=numeric_data.columns)
        # fit : Compute the mean and std to be used for later scaling.
        # transform : Perform standardization by centering and scaling
        # fit_transform : Fit to data, then transform it.

        # scaler.fit_transform(numeric_data) returns a numpy array, so we convert it back to a pandas DataFrame
        # numpy array has no columns, no index, so we need to specify them
        # convert the numpy array to a pandas DataFrame with the same columns and index as numeric_data
        # scaled_array = scaler.fit_transform(numeric_data)
        # numeric_data = pd.DataFrame(scaled_array, columns=numeric_data.columns, index=numeric_data

    correlation_matrix = numeric_data.corr()    # Compute the correlation matrix
                                                # Calculate the correlation between each pair of features(colums)
                                                # correlation_matrix is a pandas DataFrame
                                                
    np.fill_diagonal(correlation_matrix.values, 0)
    correlation_matrix.to_csv("correlation_matrix.csv")
    print("Correlation matrix saved to correlation_matrix.csv")
    # correlation_matrix.values[[range(len(correlation_matrix))]*2] = 0   # Exclude self-correlation
                                                                        # correlation_matrix.value returns a numpy array
                                                                        # len(correlation_matrix) returns the number of features(columns): if n=3
                                                                        # range(len(correlation_matrix)) returns a list of indices: [0, 1, 2]
                                                                        # [range(len(correlation_matrix))]*2 is diagonal indices: [[0, 1, 2], [0, 1, 2]]
                                                                        # correlation_matrix.values[[0, 1, 2], [0, 1, 2]] = 0
                                                                        # diagonal elements are self-correlation, so we set them to 0: (0, 0), (1, 1), (2, 2)
                                                                        # 
    return correlation_matrix.unstack().idxmax()    # Return the maximum correlation (the pair with the highest correlation)
                                                    # correlation_matrix.unstack() converts a 2D correlation matrix to a 1D Series
                                                    # idxmax() returns the index(name) of the first pair of the maximum value

# Example matrix
# matrix = np.array([
#     [1, 0.5, 0.3],
#     [0.5, 1, 0.8],
#     [0.3, 0.8, 1]
# ])
# matrix[[range(3)]*2] = 0
# correlation_matrix.values[0, 0] = 0
# correlation_matrix.values[1, 1] = 0
# correlation_matrix.values[2, 2] = 0
# [[0.0 0.5 0.3]
#  [0.5 0.0 0.8]
#  [0.3 0.8 0.0]]

# Unstacked Correlation Matrix:
# Feature1  Feature1    0.0
#           Feature2    0.8
#           Feature3    0.6
# Feature2  Feature1    0.8
#           Feature2    0.0
#           Feature3    0.7
# Feature3  Feature1    0.6
#           Feature2    0.7
#           Feature3    0.0
# dtype: float64
# idxmax() returns the index of the first maximum value: (Feature2, Feature1) or (Feature1, Feature2)

def save_standardized_data(data, output_path="standardized_data.csv"):
    exclude_columns = {"First Name", "Last Name", "Birthday", "Best Hand", "Hogwarts House"}
    numeric_data = data.drop(columns=exclude_columns, errors='ignore').select_dtypes(include=[np.number])
    numeric_data.to_csv("original_numeric_data.csv", index=False)
    print("Original numeric data saved to original_numeric_data.csv")

    scaler = StandardScaler()
    standardized_data = pd.DataFrame(
        scaler.fit_transform(numeric_data),
        columns=numeric_data.columns
    )
    standardized_data.to_csv(output_path, index=False)
    print(f"Standardized data saved to {output_path}")

    return standardized_data

def find_top_n_similar_features(data, n=5, scale_data=True):
    exclude_columns = {"Index", "First Name", "Last Name", "Birthday", "Best Hand", "Hogwarts House"}
    numeric_data = data.drop(columns=exclude_columns, errors='ignore').select_dtypes(include=[np.number])

    if scale_data:
        scaler = StandardScaler()
        numeric_data = pd.DataFrame(scaler.fit_transform(numeric_data), columns=numeric_data.columns)
    
    print("Standardized Data Preview: ")
    print(numeric_data.head())
    numeric_data.to_csv("standardized_data.csv", index=True)
    print("\n✅ Standardized data saved to standardized_data.csv")


    correlation_matrix = numeric_data.corr()
    np.fill_diagonal(correlation_matrix.values, 0)  # Set self-correlation to 0

    # Flatten the correlation matrix and sort by values
# IF you wanna see the process, comment out line:130 - 141 and uncomment line:143 - 163
    sorted_correlation = (
        correlation_matrix.unstack()
        .reset_index()
        .drop_duplicates(subset=0, keep="last")
        .rename(columns={0: "correlation", "level_0": "subject_1", "level_1": "subject_2"})
        )
    sorted_correlation = sorted_correlation[
        sorted_correlation["subject_1"] != sorted_correlation["subject_2"]
        ].sort_values(by="correlation", ascending=False)

    print("\nSorted Correlation (Descending Order):")
    print(sorted_correlation)

# sorted_correlation = (
    #     correlation_matrix.unstack()
    #     .reset_index()
    # )
    # print("\n[1] Flatten matrix:")
    # print(sorted_correlation)

    # sorted_correlation = (
    #     correlation_matrix.unstack()
    #     .reset_index()
    #     .drop_duplicates(subset=0, keep="last")
    #     .rename(columns={0: "correlation", "level_0": "subject_1", "level_1": "subject_2"})
    # )
    # sorted_correlation = sorted_correlation[
    #     sorted_correlation["subject_1"] != sorted_correlation["subject_2"]
    # ]
    # print("\n[2] Dropped Duplicated Pairs:")
    # print(sorted_correlation)
    # sorted_correlation = sorted_correlation.sort_values(by="correlation", ascending=False)
    # print("\nFinal Sorted Correlation (Descending Order):")
    # print(sorted_correlation)

# Get the top N feature pairs
    top_n_pairs = sorted_correlation.head(n)[["subject_1", "subject_2"]].values.tolist()

    return top_n_pairs

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scatter_plot.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    data = load_dataset(file_path, 'Index')
    if data is None:
        sys.exit(1)
    # save_standardized_data(data)

    # feature1, feature2 = find_most_similar_features(data, scale_data=True)
    # print(f"The most similar features are: {feature1} and {feature2}")

    # output_path = "scatter_plot.png"
    # plot_scatter(data, feature1, feature2, output_path=output_path)
    top_pairs = find_top_n_similar_features(data, n=5, scale_data=True)
    print("Top 3 feature pairs with the highest correlation:")
    for rank, (feature1, feature2) in enumerate(top_pairs, start=1):
        print(f"{rank}. {feature1} and {feature2}")

        # Generate scatter plot for each pair
        output_dir = "scatter_plots"
        output_path = os.path.join(output_dir, f"scatter_plot_top_{rank}.png")
        plot_scatter(data, feature1, feature2, output_path=output_path)
