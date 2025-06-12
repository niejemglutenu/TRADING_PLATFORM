# trading_platform/app/modeling/training_pipeline.py
import logging
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple, Any
import keras # Or from tensorflow import keras

from app.data_ingestion.db_manager import DataManager # Import your DataManager
from app.feature_engineering.strategies import FeatureEngineeringStrategy
# Example for isinstance check, or rely on feature_engineer.get_feature_names()
from app.feature_engineering.strategies import ReturnsVarCorrStrategy
from app.modeling.model_builders import create_lstm_model # Your LSTM model creation function
# Assuming create_X_Y_sequenced_for_training is moved or accessible, e.g., from a utils module
# For now, let's assume it's in a modeling_utils.py or similar if not part of this file.
# from app.modeling.modeling_utils import create_X_Y_sequenced_for_training
from app.common.constants import FEATURE_CORRELATION, FEATURE_TARGET # and others
from app.common.config import AppConfig # Assuming you have a config module for AppConfig
logger = logging.getLogger("app.modeling.training_pipeline")


# You need this function (create_X_Y_sequenced_for_training) defined or imported here
def create_X_Y_sequenced_for_training(data_df, feature_cols, target_col, window_size):
    X, Y = [], []
    if len(data_df) < window_size + 1:
        return np.array(X), np.array(Y)
    for col in feature_cols:
        if col not in data_df.columns:
            logger.error(f"Sequencing: Feature column '{col}' not found. Cols: {data_df.columns.tolist()}")
            return np.array([]), np.array([])
    if target_col not in data_df.columns:
        logger.error(f"Sequencing: Target column '{target_col}' not found. Cols: {data_df.columns.tolist()}")
        return np.array([]), np.array([])
    
    # Ensure numeric types before accessing .values
    for col in feature_cols: data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
    data_df[target_col] = pd.to_numeric(data_df[target_col], errors='coerce')
    data_df.dropna(subset=feature_cols + [target_col], inplace=True) # Drop rows if conversion failed for any feature/target

    if len(data_df) < window_size + 1: # Check again after dropna
        logger.debug(f"Data length {len(data_df)} too short after coerce/dropna for window {window_size} + 1 target.")
        return np.array(X), np.array(Y)

    feature_data_np = data_df[feature_cols].values
    target_data_np = data_df[target_col].values

    for i in range(window_size, len(data_df)): # Target is for day i, features are i-window to i-1
        sequence_x = feature_data_np[i-window_size:i, :]
        X.append(sequence_x)
        # Y value is the target for the *end* of the sequence.
        # If target is return.shift(-1), target at index `i-1` is for the actual return of day `i`.
        # So features up to `i-1` predict return of day `i`.
        Y.append(target_data_np[i-1]) 
    return np.array(X), np.array(Y)


def train_model_pipeline(
    tickers_for_training: List[str],
    index_ticker_symbol: Optional[str], # For fetching index data if strategy needs it
    training_data_start_date_str: str,
    training_data_end_date_str: str,
    model_identifier_str: str,
    feature_engineer: FeatureEngineeringStrategy,
    data_manager: DataManager, # Pass the DataManager instance
    ohlcv_table_name: str, 
    force_retrain: bool = False, # Changed to bool
    model_artifacts_base_path: Optional[str] = None # Base path for saving models
) -> Tuple[Optional[str], Optional[str], Optional[str]]:

    lstm_window_size = feature_engineer.config.get('lstm_window_size', 10)
    
    if not model_artifacts_base_path: # Get from AppConfig if not passed directly
        model_artifacts_base_path = AppConfig.get('storage.model_artifact_path', 'data/models/')
        # Ensure it's an absolute path if resolved from config
        if model_artifacts_base_path is not None and not os.path.isabs(model_artifacts_base_path):
            model_artifacts_base_path = os.path.join(AppConfig.get_project_root(), model_artifacts_base_path)
    
    try:
        if model_artifacts_base_path is not None:          
            os.makedirs(model_artifacts_base_path, exist_ok=True)
    except OSError as e_mkdir:
        logger.error(f"Failed to create model artifacts directory '{model_artifacts_base_path}': {e_mkdir}")
        return None, None, None
    

    
    safe_artifacts_path = model_artifacts_base_path if model_artifacts_base_path is not None else ""

    model_filename = os.path.join(safe_artifacts_path, f"{model_identifier_str}.keras")
    x_scaler_filename = os.path.join(safe_artifacts_path, f"{model_identifier_str}_x_scaler.gz")
    y_scaler_filename = os.path.join(safe_artifacts_path, f"{model_identifier_str}_y_scaler.gz")
    
    if not force_retrain and all(os.path.exists(f) for f in [model_filename, x_scaler_filename, y_scaler_filename]):
        logger.info(f"Model files for '{model_identifier_str}' exist at {model_artifacts_base_path}. Skipping training.")
        return model_filename, x_scaler_filename, y_scaler_filename

    logger.info(f"--- Training Model '{model_identifier_str}' using Strategy: {feature_engineer.__class__.__name__} ---")
    logger.info(f"Training Data Window: {training_data_start_date_str} to {training_data_end_date_str}")

    # 1. Fetch raw stock data for training tickers
    raw_stock_data_map = data_manager.get_data_from_db(
        tickers_list=tickers_for_training,
        start_date_str=training_data_start_date_str,
        end_date_str=training_data_end_date_str,
        table_name=ohlcv_table_name
    )
    if not raw_stock_data_map:
        logger.error("No raw stock data obtained for training. Aborting.")
        return None, None, None

    # 2. Fetch raw index data IF strategy needs it
    raw_index_df_for_transform = None
    strategy_needs_index = FEATURE_CORRELATION in feature_engineer.get_feature_names() # Or a more direct method from strategy
    if strategy_needs_index and index_ticker_symbol:
        logger.info(f"Fetching index data ({index_ticker_symbol}) for training context.")
        index_data_map = data_manager.get_data_from_db(
            tickers_list=[index_ticker_symbol],
            start_date_str=training_data_start_date_str,
            end_date_str=training_data_end_date_str,
            table_name=ohlcv_table_name
        )
        if index_data_map and index_ticker_symbol in index_data_map:
            raw_index_df_for_transform = index_data_map[index_ticker_symbol]
            if raw_index_df_for_transform.empty:
                logging.warning(f"Fetched index data for {index_ticker_symbol} is empty for training period.")
                raw_index_df_for_transform = None # Ensure it's None if empty
        else:
            logging.warning(f"Could not fetch or index data for {index_ticker_symbol} is empty for training.")
    elif strategy_needs_index and not index_ticker_symbol:
        logging.warning("Strategy indicates it needs index data, but no 'index_ticker_symbol' provided for training.")

    # 3. Process each ticker's data using the feature engineer
    processed_data_dict = {}
    required_raw_cols_for_fe = feature_engineer.get_required_raw_columns()

    for ticker_symbol, df_stock_raw_ticker in raw_stock_data_map.items():
        if df_stock_raw_ticker.empty:
            logging.warning(f"Empty raw data for ticker {ticker_symbol}. Skipping processing.")
            continue
        if not all(col in df_stock_raw_ticker.columns for col in required_raw_cols_for_fe):
            missing_cols = [col for col in required_raw_cols_for_fe if col not in df_stock_raw_ticker.columns]
            logging.warning(f"Ticker {ticker_symbol} raw data missing required columns {missing_cols} for strategy {feature_engineer.__class__.__name__}. Skipping.")
            continue
        
        transformed_df = feature_engineer.generate_features(
            df_stock_raw=df_stock_raw_ticker.copy(),
            df_index_raw=raw_index_df_for_transform.copy() if raw_index_df_for_transform is not None else None
        )
        
        if not transformed_df.empty and len(transformed_df) >= lstm_window_size + 1:
            processed_data_dict[ticker_symbol] = transformed_df
        else:
            actual_len = len(transformed_df) if not transformed_df.empty else 0
            logging.warning(f"Ticker {ticker_symbol}: Not enough data after transform ({actual_len} rows, "
                            f"need >= {lstm_window_size + 1}). Skipping.")

    if not processed_data_dict:
        logging.error("No data suitable for sequencing after processing all tickers. Aborting training.")
        return None, None, None

    # 4. Prepare data for LSTM
    all_X_list, all_Y_list = [], []
    training_feature_cols = feature_engineer.get_feature_names()
    target_col_name = feature_engineer.get_target_name()

    if not training_feature_cols:
        logging.error(f"Strategy {feature_engineer.__class__.__name__} did not define feature names. Aborting.")
        return None, None, None
    
    logger.info(f"Model training with features: {training_feature_cols} and target: {target_col_name}")

    for ticker_key, data_for_seq in processed_data_dict.items():
        if not all(col in data_for_seq.columns for col in training_feature_cols) or \
           target_col_name not in data_for_seq.columns:
            missing_f = [c for c in training_feature_cols if c not in data_for_seq.columns]
            missing_t = "None" if target_col_name in data_for_seq.columns else target_col_name
            logging.warning(f"Transformed data for {ticker_key} missing columns. Features missing: {missing_f}, Target missing: {missing_t}. Available: {data_for_seq.columns.tolist()}. Skipping.")
            continue
            
        X_stock, Y_stock = create_X_Y_sequenced_for_training(
            data_for_seq,
            feature_cols=training_feature_cols,
            target_col=target_col_name,
            window_size=lstm_window_size
        )
        if X_stock.size > 0 and Y_stock.size > 0:
            all_X_list.append(X_stock)
            all_Y_list.append(Y_stock)
    
    if not all_X_list:
        logging.error("No sequences (X,Y) created from any stock. Aborting training.")
        return None, None, None

    # 5. Combine, Scale, and Train
    X_combined = np.concatenate(all_X_list, axis=0)
    Y_combined = np.concatenate(all_Y_list, axis=0).reshape(-1, 1)
    logging.info(f"Shape of X_combined (unscaled): {X_combined.shape}, Y_combined: {Y_combined.shape}")

    num_model_features = X_combined.shape[2]
    if num_model_features != len(training_feature_cols):
        logging.error(f"CRITICAL: Feature count mismatch. X_combined has {num_model_features} features, "
                      f"strategy defined {len(training_feature_cols)} features ({training_feature_cols}). Aborting.")
        return None, None, None

    x_scaler = StandardScaler()
    X_scaled_flat = x_scaler.fit_transform(X_combined.reshape(-1, num_model_features))
    X_train_scaled = X_scaled_flat.reshape(X_combined.shape[0], lstm_window_size, num_model_features)
    
    y_scaler = StandardScaler()
    Y_train_scaled = y_scaler.fit_transform(Y_combined)

    if X_train_scaled.shape[0] == 0: # Should be caught by previous checks ideally
        logging.error("No training samples available after scaling. Aborting.")
        return None, None, None

    model_input_shape = (lstm_window_size, num_model_features)
    model = create_lstm_model(input_shape_param=model_input_shape)
    

    epochs = feature_engineer.config.get('training_epochs', AppConfig.get('model_settings.default_training_epochs', 50))
    batch_size = feature_engineer.config.get('training_batch_size', AppConfig.get('model_settings.default_batch_size', 32))

    model.fit(X_train_scaled, Y_train_scaled, epochs=epochs, batch_size=batch_size, verbose="auto", validation_split=0.1) # verbose=1 for training progress
    
    try:
        model.save(model_filename)
        joblib.dump(x_scaler, x_scaler_filename)
        joblib.dump(y_scaler, y_scaler_filename)
        logging.info(f"Saved model & scalers for '{model_identifier_str}' to '{model_artifacts_base_path}'")
    except Exception as e_save:
        logging.error(f"Error saving model/scalers for {model_identifier_str}: {e_save}", exc_info=True)
        return None, None, None

    return model_filename, x_scaler_filename, y_scaler_filename