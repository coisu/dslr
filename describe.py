import sys
import csv
import numpy as np
from utils import load_dataset

def calculate_mean(data):
    return sum(data) / len(data)

def calculate_variance(data, mean):
    return sum((x - mean) ** 2 for x in data) / (len(data) - 1) if len(data) > 1 else 0

def calculate_std(variance):
    return variance ** 0.5  # math.sqrt(variance)

def calculate_percentile(data, p):
    k = (len(data) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c < len(data):
        return data[f] + (k - f) * (data[c] - data[f])
    else:
        return data[f]

def is_float(value):
    """is numeric data"""
    try:
        float(value)
        return True
    except ValueError:
        return False

def calculate_statistics(data):
    exclude_columns = {"First Name", "Last Name", "Birthday", "Best Hand", "Hogwarts House"}
    statistics = {}

    for column in data.columns:
        if column in exclude_columns or not np.issubdtype(data[column].dtype, np.number):
            continue

        col_data = data[column].dropna().sort_values() # .dropna() filters out missing values e.g. NaN
        # data = pandas.DataFrame
        # data[column] = pandas.Series
        if not col_data.empty:
            mean = calculate_mean(col_data)
            variance = calculate_variance(col_data, mean)
            std = calculate_std(variance)
            min = col_data[0]
            max = col_data.iloc[-1] # pandas.Series cannot access with negative index
            p25 = calculate_percentile(col_data, 25)
            p50 = calculate_percentile(col_data, 50)
            p75 = calculate_percentile(col_data, 75)

            statistics[column] = {
                'count': len(col_data),
                'mean': mean,
                'std': std,
                'min': min,
                '25%': p25,
                '50%': p50,
                '75%': p75,
                'max': max
            }

    return statistics

def print_statistics(statistics):
    print(f"{'Feature':<30} {'Count':<10} {'Mean':<10} {'Std':<10} {'Min':<10} {'25%':<10} {'50%':<10} {'75%':<10} {'Max':<10}")
    print("-" * 120)
    for feature, stats in statistics.items():  # 열 이름(Feature)을 키로 사용
        print(f"{feature:<30} {stats['count']:<10.2f} {stats['mean']:<10.2f} {stats['std']:<10.2f} {stats['min']:<10.2f} {stats['25%']:<10.2f} {stats['50%']:<10.2f} {stats['75%']:<10.2f} {stats['max']:<10.2f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 describe.py <dataset.csv>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    data = load_dataset(file_path, 'Index') # pd.DataFrame
    if data is None:
        print("Failed to load the dataset.")
        sys.exit(1)
    statistics = calculate_statistics(data) # dict
    print_statistics(statistics)
