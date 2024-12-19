import sys
import csv
import numpy as np

def load_csv(file_path):
    """load CSV, return data"""
    try:
        with open(file_path, 'r') as f:
            rdr = csv.reader(f)
            headers = next(rdr) # first row is headers
            data = []
            for row in rdr:
                data.append(row)
        # print("data[0]: ", data[0])
            # rows = list(rdr)  # turn into list 
            # headers, data = rows[0], rows[1:] # first row is headers, rest is data
        return headers, data
    
    except FileNotFoundError:
        print(f"Error: File not found - '{file_path}'. Please check the file path.")
        return None, None

    except PermissionError:
        print(f"Error: Permission denied for file - '{file_path}'.")
        return None, None

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None

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

def calculate_statistics(headers, data):
    exclude_columns = {"First Name", "Last Name", "Birthday", "Best Hand", "Hogwarts House"}
    statistics = {}

    for col, header in enumerate(headers):
        if header in exclude_columns: # possible to exist 'NaN', 'INF' or 'Infinity' as string
            continue
        col_data = [float(row[col]) for row in data if is_float(row[col])]
        col_data = [x for x in col_data if not np.isnan(x)]  # filter out NaN values   
        
        if col_data:
            col_data.sort()  # for percentile calculation
            count = len(col_data)
            mean = calculate_mean(col_data)
            variance = calculate_variance(col_data, mean)
            std = calculate_std(variance)
            min = col_data[0]
            max = col_data[-1]
            p25 = calculate_percentile(col_data, 25)
            p50 = calculate_percentile(col_data, 50)
            p75 = calculate_percentile(col_data, 75)

            statistics[header] = {
                'count': count,
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
    headers, data = load_csv(file_path)
    statistics = calculate_statistics(headers, data)
    print_statistics(statistics)
