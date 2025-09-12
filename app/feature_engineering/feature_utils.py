import logging
import numpy as np
import pandas as pd
from typing import List, Tuple

def create_X_Y_sequenced_for_training(data_df: pd.DataFrame, feature_cols: List[str], target_col: str, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    X, Y = [], []
    if len(data_df) < window_size + 1: return np.array(X), np.array(Y)
    
    for col in feature_cols + [target_col]:
        if col not in data_df.columns:
            logging.error(f"Sequencing Error: Column '{col}' not found in DataFrame. Available: {data_df.columns.tolist()}")
            return np.array([]), np.array([])
        try:
            data_df[col] = pd.to_numeric(data_df[col], errors='raise')
        except Exception as e:
            logging.error(f"Sequencing Error: Column '{col}' cannot be converted to numeric: {e}")
            return np.array([]), np.array([])

    feature_data_np = data_df[feature_cols].values
    target_data_np = data_df[target_col].values 

    for i in range(window_size, len(data_df)): 
        sequence_x = feature_data_np[i - window_size:i, :]
        X.append(sequence_x)

        Y.append(target_data_np[i-1]) 
    return np.array(X), np.array(Y)

