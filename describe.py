import sys
import csv
import numpy as np

def load_csv(file_path):
    """load CSV, return data"""
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # 첫 번째 행은 헤더
        data = []
        for row in reader:
            data.append(row)
    return headers, data

def is_float(value):
    """check if is able to turn into float"""
    try:
        float(value)
        return True
    except ValueError:
        return False

def calculate_statistics(data):
    """cal statistics numeric data"""
    statistics = {}
    for column in range(len(data[0])):
        column_data = [float(row[column]) for row in data if is_float(row[column])]
        if column_data:
            statistics[column] = {
                'count': len(column_data),
                'mean': np.mean(column_data),
                'std': np.std(column_data, ddof=1),
                'min': np.min(column_data),
                '25%': np.percentile(column_data, 25),
                '50%': np.percentile(column_data, 50),
                '75%': np.percentile(column_data, 75),
                'max': np.max(column_data)
            }
    return statistics

def print_statistics(headers, statistics):
    print(f"{'Feature':<15} {'Count':<10} {'Mean':<10} {'Std':<10} {'Min':<10} {'25%':<10} {'50%':<10} {'75%':<10} {'Max':<10}")
    for column, stats in statistics.items():
        print(f"{headers[column]:<15} {stats['count']:<10.2f} {stats['mean']:<10.2f} {stats['std']:<10.2f} {stats['min']:<10.2f} {stats['25%']:<10.2f} {stats['50%']:<10.2f} {stats['75%']:<10.2f} {stats['max']:<10.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 describe.py <dataset.csv>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    headers, data = load_csv(file_path)
    statistics = calculate_statistics(data)
    print_statistics(headers, statistics)
