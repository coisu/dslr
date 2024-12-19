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

def is_float(value):
    """check if is able to turn into numeric value"""
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
        print(f"Column data: {col_data}\n")  # 현재 열의 값 출력
        if col_data:
            statistics[header] = {
                'count': len(col_data),
                'mean': np.mean(col_data),  # average
                'std': np.std(col_data, ddof=1) if len(col_data) > 1 else 0, # standard deviation, ddof=1: Sample Standard Deviation.  improves the accuracy of the estimate by compensating for the bias caused by using sample data. The degrees of freedom adjustment helps correct the issue of underestimation.
                'min': np.min(col_data),
                '25%': np.percentile(col_data, 25),
                '50%': np.percentile(col_data, 50),
                '75%': np.percentile(col_data, 75),
                'max': np.max(col_data)
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
