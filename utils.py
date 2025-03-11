import pandas as pd

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
