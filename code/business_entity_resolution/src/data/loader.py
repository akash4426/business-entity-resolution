import pandas as pd
from typing import Optional

def load_source(filepath: str) -> pd.DataFrame:
    """
    Load a source TSV file safely.
    Preserves strings as strings and handles empty fields without crashing.
    """
    df = pd.read_csv(filepath, sep='\t', dtype=str, keep_default_na=False)
    # Standardize missing values to empty strings just in case
    df = df.fillna('')
    return df

def load_ground_truth(filepath: str) -> pd.DataFrame:
    """
    Load ground truth TSV file safely.
    """
    df = pd.read_csv(filepath, sep='\t', dtype=str, keep_default_na=False)
    df = df.fillna('')
    return df
