import pandas as pd

def load_training_data(file_path = "data/training_data.xlsx", sheet_name=0):
    """
    Loads data from an Excel file into a pandas DataFrame.
    
    Args:
    file_path (str): The path to the Excel file to load.
    sheet_name (str, int, list, or None, optional): Specifies which sheet should be loaded. Default is the first sheet.
    
    Returns:
    pd.DataFrame: A DataFrame containing the loaded data.
    """
    # Load the Excel file
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    return df