import sys
import csv
import numpy as np
from utils import load_dataset, calculate_mean, calculate_variance, calculate_std, calculate_percentile, is_float


def calculate_statistics(data):
    exclude_columns = {"Index", "First Name", "Last Name", "Birthday", "Best Hand", "Hogwarts House"}
    statistics = {}

    for column in data.columns:
        if column in exclude_columns or not np.issubdtype(data[column].dtype, np.number):
            continue

        col_data = data[column].dropna().tolist() # .dropna() filters out missing values e.g. NaN
        col_data.sort()
        # data = pandas.DataFrame
        # data[column] = pandas.Series
        if col_data:
            mean = calculate_mean(col_data)
            variance = calculate_variance(col_data, mean)
            std = calculate_std(variance)
            min_val = col_data[0]
            max_val = col_data[-1] # pandas.Series cannot access with negative index
            p25 = calculate_percentile(col_data, 25)
            p50 = calculate_percentile(col_data, 50)
            p75 = calculate_percentile(col_data, 75)

            statistics[column] = {
                'count': len(col_data),
                'mean': mean,
                'std': std,
                'min': min_val,
                '25%': p25,
                '50%': p50,
                '75%': p75,
                'max': max_val
            }

    return statistics

def print_statistics(statistics):
    print(f"{'Feature':<30} {'Count':>10} {'Mean':>10} {'Std':>10} {'Min':>10} {'25%':>10} {'50%':>10} {'75%':>10} {'Max':>10}")
    print("-" * 120)
    for feature, stats in statistics.items():
        print(f"{feature:<30} {stats['count']:>10.2f} {stats['mean']:>10.2f} {stats['std']:>10.2f} {stats['min']:>10.2f} {stats['25%']:>10.2f} {stats['50%']:>10.2f} {stats['75%']:>10.2f} {stats['max']:>10.2f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 describe.py <dataset.csv>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    print(f"\n\nCurrent File :: {file_path}\n\n")
    data = load_dataset(file_path, 'Index') # pd.DataFrame
    if data is None:
        print("Failed to load the dataset.")
        sys.exit(1)
    statistics = calculate_statistics(data) # dict
    print_statistics(statistics)

    # for debugging
    from describe import calculate_statistics

    print("\n pandas original describe:")
    print(data.describe().T)