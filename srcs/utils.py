import pandas as pd
import numpy as np
from typing import Iterable

def load_dataset(file_path, index_col=None):
    """
    Load a CSV file and return it as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path, index_col=index_col)
        return data

    except FileNotFoundError:
        print(f"Error: File not found - '{file_path}'. Please check the file path.")
        return None

    except pd.errors.EmptyDataError:
        print(f"Error: The file - '{file_path}' is empty.")
        return None

    except pd.errors.ParserError:
        print(f"Error: The file - '{file_path}' contains invalid data.")
        return None

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def manual_sum(values):
    total = 0
    for v in values:
        total += v
    return total

def calculate_mean(data: Iterable[float]) -> float:
    return manual_sum(data) / len(data)

def calculate_variance(data, mean):
    return manual_sum((x - mean) ** 2 for x in data) / (len(data) - 1) if len(data) > 1 else 0

def calculate_std(variance):
    return variance ** 0.5  # math.sqrt(variance)

def calculate_percentile(data, p):
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c < len(sorted_data):
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
    else:
        return sorted_data[f]

def is_float(value):
    """is numeric data"""
    try:
        float(value)
        return True
    except ValueError:
        return False

def compute_manual_correlation(data):
    correlation_results = {}
    features = data.columns
    
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            feature1, feature2 = features[i], features[j]
            x, y = data[feature1].values, data[feature2].values

            # NaN filtering
            valid = ~np.isnan(x) & ~np.isnan(y)     # ~ : NOT, 
            x_valid, y_valid = x[valid], y[valid]   # valid subset by NumPy 'Boolean Indexing'

            # valid : numpy.ndarray (bool dtype)
            # x_valid, y_valid : numpy.ndarray (float dtype)

            mean_x, mean_y = calculate_mean(x_valid), calculate_mean(y_valid)
            # numerator = manual_sum((x - mean_x) * (y - mean_y))
            numerator = manual_sum((x_valid - mean_x) * (y_valid - mean_y)) # vectorized operations for numpy.ndarray
            denominator_x = manual_sum((a - mean_x) ** 2 for a in x_valid)
            denominator_y = manual_sum((b - mean_y) ** 2 for b in y_valid)
            denominator = (denominator_x * denominator_y) ** 0.5
            correlation = numerator / denominator if denominator != 0 else 0
            correlation_results[(feature1, feature2)] = correlation
    
    return correlation_results