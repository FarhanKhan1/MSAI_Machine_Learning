import pandas as pd
import numpy as np


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    return pd.read_csv(file_path)

def drop_unnecessary_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    """
    Drop unnecessary columns from the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns_to_drop (list): List of column names to drop.

    Returns:
    pd.DataFrame: The DataFrame after dropping the specified columns.
    """
    return df.drop(columns=columns_to_drop, errors='ignore')

def preprocess_data()-> pd.DataFrame:
    """
    This function preprocess the raw data and returns a cleaned DataFrame.
    The exact steps are following


    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """
    to_be_dropped = ['Id','1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'BsmtFinSF1', 'BsmtFinSF2','BsmtUnfSF','GarageArea',
    'GarageYrBlt', 'YearRemodAdd', 'LotFrontage', 'ExterQual', 'BsmtQual', 'KitchenQual', 'Utilities', 'Street',
    'Condition2', 'RoofMatl', 'PoolArea', 'PoolQC', '3SsnPorch', 'LandSlope', 'Alley', 'Heating',"BsmtFullBath", 
    "BsmtHalfBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", 
    "ScreenPorch", "MiscVal", "MoSold", "YrSold", "TotRmsAbvGrd","MSSubClass", "MSZoning", "LandContour",
    "Foundation", "PavedDrive", "SaleType", "HouseStyle","LotShape"]    

    raw_dataset_path = "../data/raw_dataset/train.csv" 
    df = load_data(raw_dataset_path)
    return drop_unnecessary_columns(df, to_be_dropped)