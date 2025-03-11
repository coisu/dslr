import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils import load_dataset

def compute_manual_correlation(data):
    correlation_results = {}
    features = data.columns
    
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            feature1, feature2 = features[i], features[j]
            x, y = data[feature1].values, data[feature2].values
            mean_x, mean_y = np.mean(x), np.mean(y)
            numerator = np.sum((x - mean_x) * (y - mean_y))
            denominator = np.sqrt(np.sum((x - mean_x) ** 2) * np.sum((y - mean_y) ** 2))
            correlation = numerator / denominator if denominator != 0 else 0
            correlation_results[(feature1, feature2)] = correlation
    
    return correlation_results

def plot_pair(data, output_path="pair_plot.png"):
    """
    Generates a pair plot (scatter plot matrix) for numerical features in the dataset.
    Saves the output as an image file.
    """
    if os.path.dirname(output_path):  # Ensure it's not empty
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    exclude_columns = {"Index", "First Name", "Last Name", "Birthday", "Best Hand", "Hogwarts House"}
    numeric_data = data.drop(columns=exclude_columns, errors="ignore").select_dtypes(include=['number'])
    
    # Handle missing values (NaN): Fill NaN values with the mean of the column
    numeric_data = numeric_data.fillna(numeric_data.mean())  # Fill missing values with column means
    # Scale numbers
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(numeric_data), columns=numeric_data.columns)

    print("Scaled data oreview:")
    print(scaled_data.head())

    # Ensure "Hogwarts House" is treated as a categorical variable
    if "Hogwarts House" in data.columns:
        scaled_data["Hogwarts House"] = data["Hogwarts House"]
    
    # Compute manual correlation
    correlation_results = compute_manual_correlation(scaled_data.drop(columns=["Hogwarts House"], errors="ignore"))
    sorted_correlation = sorted(correlation_results.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Print top correlated feature pairs
    print("\nTop feature correlations:")
    for (feature1, feature2), corr in sorted_correlation[:5]:
        print(f"{feature1} & {feature2}: {corr:.3f}")
    
    # Pair plot with seaborn
    pair_plot = sns.pairplot(scaled_data, hue="Hogwarts House", palette={"Gryffindor": "red", 
                                                                         "Slytherin": "green", 
                                                                         "Ravenclaw": "blue", 
                                                                         "Hufflepuff": "orange"})
    
    plt.savefig(output_path, dpi=300)
    print(f"Pair plot saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pair_plot.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]

    data = load_dataset(file_path, 'Index')

    if data is None:
        sys.exit(1)
    
    output_path = "pair_plot.png"
    plot_pair(data, output_path)
