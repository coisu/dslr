import sys
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import load_dataset, compute_manual_correlation

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

    correlation_results = compute_manual_correlation(numeric_data)
    sorted_correlation = sorted(correlation_results.items(), key=lambda x: abs(x[1]), reverse=True)

    print("\nSorted Correlation (Descending Order):")
    for (subject_1, subject_2), corr in sorted_correlation[:n]:
        print(f"{subject_1} & {subject_2} → correlation: {corr:.4f}")

# Get the top N feature pairs
    # top_n_pairs = sorted_correlation.head(n)[["subject_1", "subject_2"]].values.tolist()
    top_n_pairs = [(subject_1, subject_2) for (subject_1, subject_2), _ in sorted_correlation[:n]]

    return top_n_pairs, numeric_data

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
    top_pairs, standardized_data = find_top_n_similar_features(data, n=5, scale_data=True)
    print("Top 5 feature pairs with the highest correlation:")
    for rank, (feature1, feature2) in enumerate(top_pairs, start=1):
        print(f"{rank}. {feature1} and {feature2}")

        # Generate scatter plot for each pair
        output_dir = "scatter_plots"
        output_path = os.path.join(output_dir, f"scatter_plot_top_{rank}.png")
        plot_scatter(standardized_data, feature1, feature2, output_path=output_path)
