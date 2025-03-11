import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
from utils import load_dataset

def calculate_variance_std(data, columns, group_column="Hogwarts House"):
    results = []
    for column in columns:
        group_stats = data.groupby(group_column, observed=False)[column].agg(['var', 'std']).reset_index()
        group_stats['Course'] = column
        results.append(group_stats)
    return pd.concat(results, ignore_index=True)

def normalize_data(data, columns):
    normalized_data = data.copy()
    for column in columns:
        min_val = data[column].min()
        max_val = data[column].max()
        normalized_data[column] = (data[column] - min_val) / (max_val - min_val)
    return normalized_data

def plot_std_line_graph(variance_std_data, output_dir="histograms", output_file="std_line_graph.png"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(12, 6))
    
    sns.lineplot(
        data=variance_std_data,
        x="Course",
        y="std",
        hue="Hogwarts House",
        palette={"Gryffindor": "red", "Slytherin": "green", "Ravenclaw": "blue", "Hufflepuff": "orange"},
        marker="o"  # Add markers to the line
    )

    plt.title("Standard Deviation of Scores by Course and House", fontsize=14)
    plt.xlabel("Course", fontsize=12)
    plt.ylabel("Standard Deviation", fontsize=12)
    plt.xticks(rotation=45, ha="right")         # Rotate x-axis labels for readability
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "std_line_graph.png")
    plt.savefig(output_file)
    print(f"Saved standard deviation line graph to {output_file}")

    plt.close()


def plot_histogram(data, columns, hue_column="Hogwarts House", output_dir="histograms"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("Hogwarts House distribution:")
    print(data[hue_column].value_counts())
    
    if data[hue_column].dtype != "category":
        data[hue_column] = data[hue_column].astype("category")
    
    house_colors = {
        "Gryffindor": "red",
        "Slytherin": "green",
        "Ravenclaw": "blue",
        "Hufflepuff": "orange"
    }

    for column in columns:
        col_data = data[[column, hue_column]].dropna()

        if col_data[hue_column].nunique() > 1 and not col_data[column].isnull().all():
            plt.figure(figsize=(10, 6))
            sns.histplot(
                data=col_data,
                x=column,
                hue=hue_column,
                kde=True,
                element="step",
                stat="density",
                common_norm=False,
                palette=house_colors
            )
            plt.title(f"Score Distribution for {column}")
            plt.xlabel(column)
            plt.ylabel("Density")

            handles, labels = plt.gca().get_legend_handles_labels()
            
            if handles and labels:
                plt.legend(handles=handles, labels=labels, title=hue_column, loc="upper right")
            else:
                custom_labels = list(house_colors.keys())
                custom_colors = [house_colors[label] for label in custom_labels]
                patches = [plt.Line2D([0], [0], color=color, lw=4) for color in custom_colors]
                plt.legend(handles=patches, labels=custom_labels, title=hue_column, loc="upper right")

            # handles, labels = plt.gca().get_legend_handles_labels()
            # if not labels:
            #     labels = data[hue_column].unique()
            #     plt.legend(title=hue_column, labels=labels, loc="upper right")
            # labels = plt.gca().get_legend_handles_labels()[1]  # labels만 가져옴
            # if not labels:
            #     labels = data[hue_column].unique()
            #     plt.legend(title=hue_column, labels=labels, loc="upper right")
            
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()

            output_file = os.path.join(output_dir, f"{column}_histogram.png")
            plt.savefig(output_file)
            print(f"Saved histogram for {column} to {output_file}")
        else:
            print(f"Skipping {column}, insufficient data for histogram.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 histogram.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    data = load_dataset(dataset_path, 'Index')

    if data is None:
        print("Failed to load the dataset.")
        sys.exit(1)

    # selected_plot = ["Potions", "Transfiguration", "Charms", "Flying", "Herbology"]
    numeric_columns = [col for col in data.columns if data[col].dtype in ['int64', 'float64'] and col != 'Index']
    data = data.dropna(subset=["Hogwarts House"])
    plot_histogram(data, numeric_columns, output_dir="histograms")
    
    variance_std_data = calculate_variance_std(data, numeric_columns)

    normalized_data = normalize_data(data, numeric_columns)
    variance_std_data_normalized = calculate_variance_std(normalized_data, numeric_columns)
    plot_std_line_graph(variance_std_data_normalized, output_file="std_line_graph_normalized.png")