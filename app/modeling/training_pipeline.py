import logging
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple, Dict
from pathlib import Path

from app.data_ingestion.db_manager import DataManager
from app.feature_engineering.strategies import FeatureEngineeringStrategy
from app.modeling.model_builders import LSTMModel

logger = logging.getLogger("app.modeling.training_pipeline")

def train_model_pipeline(
    tickers_for_training: List[str],
    index_ticker_symbol: Optional[str],
    training_data_start_date_str: str,
    training_data_end_date_str: str,
    model_identifier_str: str,
    feature_engineer: FeatureEngineeringStrategy,
    data_manager: DataManager,
    ohlcv_table_name: str,
    force_retrain: bool = False,
    model_artifacts_base_path: Optional[str] = None,
    model: str = 'LSTM_Shuffle',
    epochs: int = 50,
    batch_size: int = 32,
    prediction_horizon: int = 2
) -> Optional[str]:

    try:
        if not model_artifacts_base_path:
            raise ValueError("model_artifacts_base_path cannot be None")
        
        model_dir = os.path.join(model_artifacts_base_path, model_identifier_str)
        
        if Path(model_dir, "model.keras").exists() and not force_retrain:
            logger.info(f"Model already exists at {model_dir}, skipping training.")
            return model_dir

        logger.info(f"Preparing training data for tickers: {tickers_for_training}")

        # 1. Fetch all required data in one go
        all_tickers_to_fetch = list(set(tickers_for_training + ([index_ticker_symbol] if index_ticker_symbol else [])))
        historical_data = data_manager.get_data_from_db(
            all_tickers_to_fetch,
            training_data_start_date_str,
            training_data_end_date_str,
            ohlcv_table_name
        )
        index_df = historical_data.get(index_ticker_symbol)

        # 2. Generate features for each stock and collect them
        all_feature_dfs = []
        for ticker in tickers_for_training:
            stock_df = historical_data.get(ticker)
            if stock_df is None or stock_df.empty:
                logger.warning(f"No data for training ticker {ticker}, skipping.")
                continue
            
            # Remove the 'ticker' column if it exists to prevent duplication
            if 'ticker' in stock_df.columns:
                stock_df = stock_df.drop(columns=['ticker'])
            
            feature_df = feature_engineer.generate_features(stock_df, df_index_raw=index_df, prediction_horizon=prediction_horizon)
            # It's important to have the Ticker column for grouping later
            feature_df['Ticker'] = ticker 
            all_feature_dfs.append(feature_df)

        if not all_feature_dfs:
            logger.error("No feature data could be generated for any training tickers.")
            return None

  
        combined_feature_df = pd.concat(all_feature_dfs)
        logger.info(f"Combined feature DataFrame for training has shape: {combined_feature_df.shape}")
  
        logger.info(f"Shape after concatenation (no global de-duplication): {combined_feature_df.shape}")
        feature_cols = feature_engineer.get_feature_names()
        target_col = feature_engineer.get_target_name()
        
        cols_for_training = feature_cols + [target_col, 'Ticker']
        
        final_df_for_training = combined_feature_df.dropna(subset=cols_for_training).copy()

        missing_cols = [c for c in cols_for_training if c not in combined_feature_df.columns]
        if missing_cols:
            logger.error(f"FATAL: The following required columns are missing from the combined DataFrame: {missing_cols}")
            return None
        print("###########################final_df_for_training########################")
        print(final_df_for_training.head())
        print(final_df_for_training.columns)

        # Pass this clean DataFrame to the training function
        success, saved_path = _perform_training_on_combined_df(
            combined_df=final_df_for_training, # Pass the clean df
            feature_engineer=feature_engineer,
            model_save_path=model_dir,
            model=model,
            epochs=epochs,
            batch_size=batch_size
        )
        
        if not success:
            logger.error(f"Failed to train model '{model_identifier_str}'.")
            return None

        logger.info(f"Successfully trained and saved model '{model_identifier_str}' to '{saved_path}'")
        return saved_path

    except Exception as e:
        logger.error(f"Error in train_model_pipeline: {e}", exc_info=True)
        return None


def _check_data_integrity(df, name="DataFrame"):
    logger.debug(f"{name} - Shape: {df.shape}, Index unique: {df.index.is_unique}")
    if df.index.has_duplicates:
        duplicates = df.index[df.index.duplicated()].tolist()
        logger.debug(f"{name} - Duplicate indices: {duplicates[:5]}...")  # Show first 5
    return df.index.is_unique

def _debug_dataframe_info(df, name="DataFrame"):
    logger.debug(f"{name} - Shape: {df.shape}")
    logger.debug(f"{name} - Columns: {df.columns.tolist()}")
    logger.debug(f"{name} - Index type: {type(df.index)}")
    logger.debug(f"{name} - First few rows:")
    logger.debug(f"{df.head(3)}")

def _create_scaled_sequences_from_group(group, feature_cols, target_col, window_size):
    if len(group) < window_size + 1:
        return [], []

    logger.debug(f"Feature columns: {feature_cols}")
    logger.debug(f"Target column: {target_col}")
    logger.debug(f"Group columns: {group.columns.tolist()}")
    logger.debug(f"Group shape: {group.shape}")


    if group.index.has_duplicates:
        logger.warning(f"Duplicate timestamps found in group. Shape before: {group.shape}")
        # Keep the first occurrence of each duplicated timestamp
        group = group.loc[~group.index.duplicated(keep='first')]
        logger.warning(f"Shape after de-duplication: {group.shape}")
    
    group = group.sort_index()
    
    group = group.reset_index().set_index('timestamp')
    
    if group.index.has_duplicates:
        logger.error(f"CRITICAL: Still have duplicate timestamps in _create_scaled_sequences_from_group")
        return [], []
    
    _check_data_integrity(group, "Group after cleaning")
    # ======================================================================

    cols_for_expanding = feature_cols + [target_col]
    
    if target_col in feature_cols:
        logger.error(f"Target column '{target_col}' is also in feature columns: {feature_cols}")
        logger.error(f"This will cause conflicts. Please check the feature engineering strategy.")
        return [], []
    
    clean_df = group[cols_for_expanding].copy()
    
    missing_cols = [col for col in cols_for_expanding if col not in group.columns]
    if missing_cols:
        logger.error(f"Missing columns in group: {missing_cols}")
        logger.error(f"Available columns: {group.columns.tolist()}")
        return [], []
    
    if clean_df.index.has_duplicates:
        logger.error(f"CRITICAL: Clean DataFrame still has duplicate indices")
        return [], []
    
    _check_data_integrity(clean_df, "Clean DataFrame")
    _debug_dataframe_info(clean_df, "Clean DataFrame")
    
    logger.debug(f"Data types of clean_df: {clean_df.dtypes.to_dict()}")
    logger.debug(f"Data types of feature_cols: {clean_df[feature_cols].dtypes.to_dict()}")
    
    target_data = clean_df[target_col]
    if isinstance(target_data, pd.DataFrame):
        logger.error(f"Target column '{target_col}' is returning a DataFrame with columns: {target_data.columns.tolist()}")
        logger.error(f"This suggests there's a column name conflict. Available columns: {clean_df.columns.tolist()}")
        return [], []
    else:
        logger.debug(f"Data type of target_col: {target_data.dtype}")
    
    expanding_mean = pd.DataFrame(index=clean_df.index)
    expanding_std = pd.DataFrame(index=clean_df.index)
    
    for col in cols_for_expanding:
        logger.debug(f"Processing column: {col}")
        col_series = clean_df[col]
        expanding_mean[col] = col_series.expanding(min_periods=window_size).mean()
        expanding_std[col] = col_series.expanding(min_periods=window_size).std()
        logger.debug(f"Column {col} - expanding_mean shape: {expanding_mean[col].shape}, expanding_std shape: {expanding_std[col].shape}")
        logger.debug(f"Column {col} - expanding_mean type: {type(expanding_mean[col])}, expanding_std type: {type(expanding_std[col])}")
    
    _check_data_integrity(expanding_mean, "Expanding Mean")
    _check_data_integrity(expanding_std, "Expanding Std")
    _debug_dataframe_info(expanding_mean, "Expanding Mean")
    _debug_dataframe_info(expanding_std, "Expanding Std")

    scaled_group = pd.DataFrame(index=clean_df.index)
    
    for col in feature_cols:
        try:
            logger.debug(f"Scaling column: {col}")
            logger.debug(f"clean_df[{col}] type: {type(clean_df[col])}, shape: {clean_df[col].shape}")
            logger.debug(f"expanding_mean[{col}] type: {type(expanding_mean[col])}, shape: {expanding_mean[col].shape}")
            logger.debug(f"expanding_std[{col}] type: {type(expanding_std[col])}, shape: {expanding_std[col].shape}")
            
            scaled_group[f"{col}_scaled"] = (clean_df[col] - expanding_mean[col]) / (expanding_std[col] + 1e-6)
            logger.debug(f"Successfully scaled column: {col}")
        except Exception as e:
            logger.error(f"Error scaling column {col}: {e}")
            logger.error(f"clean_df[{col}] shape: {clean_df[col].shape}")
            logger.error(f"expanding_mean[{col}] shape: {expanding_mean[col].shape}")
            logger.error(f"expanding_std[{col}] shape: {expanding_std[col].shape}")
            return [], []
    
    try:
        logger.debug(f"Scaling target column: {target_col}")
        logger.debug(f"clean_df[{target_col}] type: {type(clean_df[target_col])}, shape: {clean_df[target_col].shape}")
        logger.debug(f"expanding_mean[{target_col}] type: {type(expanding_mean[target_col])}, shape: {expanding_mean[target_col].shape}")
        logger.debug(f"expanding_std[{target_col}] type: {type(expanding_std[target_col])}, shape: {expanding_std[target_col].shape}")
        
        scaled_group[f"{target_col}_scaled"] = (clean_df[target_col] - expanding_mean[target_col]) / (expanding_std[target_col] + 1e-6)
        logger.debug(f"Successfully scaled target column: {target_col}")
    except Exception as e:
        logger.error(f"Error scaling target column {target_col}: {e}")
        logger.error(f"clean_df[{target_col}] shape: {clean_df[target_col].shape}")
        logger.error(f"expanding_mean[{target_col}] shape: {expanding_mean[target_col].shape}")
        logger.error(f"expanding_std[{target_col}] shape: {expanding_std[target_col].shape}")
        return [], []
    
    scaled_group.dropna(inplace=True)
    if len(scaled_group) < window_size + 1:
        return [], []

    feature_data = scaled_group[[f"{c}_scaled" for c in feature_cols]].values
    target_data = scaled_group[[f"{target_col}_scaled"]].values
    
    X_sequences, y_sequences = [], []
    for i in range(len(feature_data) - window_size):
        X_sequences.append(feature_data[i:(i + window_size)])
        y_sequences.append(target_data[i + window_size])
        
    return X_sequences, y_sequences

def _perform_training_on_combined_df(
    combined_df: pd.DataFrame,
    feature_engineer: FeatureEngineeringStrategy,
    model_save_path: str,
    model: str = 'LSTM_Shuffle',
    epochs: int = 50,
    batch_size: int = 32
) -> Tuple[bool, Optional[str]]:
    """
    Internal helper with aggressive diagnostic cleaning to resolve the duplicate index error.
    """
    try:
        feature_cols = feature_engineer.get_feature_names()
        target_col = feature_engineer.get_target_name()
        
        # Get window size from feature engineer config, or use default
        window_size = feature_engineer.config.get('lstm_window_size', 10)
        
        # If not in feature engineer config, try to get from model settings
        if window_size == 10:  # This means it's the default
            try:
                from app.common.config import AppConfig
                config = AppConfig.get_instance()
                window_size = config.get('model_settings', {}).get('default_lstm_window_size', 10)
                logger.info(f"Using window size from model settings: {window_size}")
            except Exception as e:
                logger.warning(f"Could not get window size from config, using default 10: {e}")
        
        logger.info(f"Using window size: {window_size}")
        logger.info(f"Feature columns: {feature_cols}")
        logger.info(f"Target column: {target_col}")
        logger.info(f"Combined DataFrame columns: {combined_df.columns.tolist()}")
        logger.info(f"Combined DataFrame shape: {combined_df.shape}")
        logger.info(f"Combined DataFrame index unique: {combined_df.index.is_unique}")
        
        logger.info(f"Final feature columns: {feature_cols}")
        logger.info(f"Final target column: {target_col}")
        logger.info(f"Columns for expanding calculations: {feature_cols + [target_col]}")

        
        if combined_df.empty:
            logger.error("Initial combined DataFrame is empty.")
            return False, None

        logger.info(f"Combined DataFrame columns: {combined_df.columns.tolist()}")
        logger.info(f"Combined DataFrame shape: {combined_df.shape}")
        logger.info(f"Combined DataFrame index unique: {combined_df.index.is_unique}")

        if model in ['LSTM_Shuffle', 'LSTM_NoShuffle']:
            
            all_X, all_y_ticker = {}, {}
            for ticker, group in combined_df.groupby('Ticker'):
                
         
                initial_shape = group.shape
                logger.debug(f"Processing ticker {ticker}, initial shape: {initial_shape}")
                
            
                if 'ticker' in group.columns:
                    group = group.drop(columns=['ticker'])
                    logger.debug(f"Removed duplicate 'ticker' column for {ticker}")
                
                group = group.sort_index()
                
                if group.index.has_duplicates:
                    logger.warning(f"DUPLICATE TIMESTAMPS DETECTED for ticker {ticker}. Shape before: {initial_shape}")
                    # Keep the first occurrence of each duplicated timestamp
                    group = group.loc[~group.index.duplicated(keep='first')]
                    logger.warning(f"Shape after de-duplication for {ticker}: {group.shape}")
                
                # Step 4: CRITICAL - Reset the index to ensure no duplicate index issues
                # This is the key fix - we reset the index to make sure pandas doesn't have any
                # internal issues with the index
                group = group.reset_index().set_index('timestamp')
                
                # Step 5: Final verification - check again for duplicates
                if group.index.has_duplicates:
                    logger.error(f"CRITICAL: Still have duplicate timestamps after de-duplication for {ticker}")
                    # Force remove duplicates one more time
                    group = group.loc[~group.index.duplicated(keep='first')]
                    logger.warning(f"Final shape after forced de-duplication for {ticker}: {group.shape}")
                
                # Step 6: Debug logging
                logger.debug(f"Final group for {ticker}: shape={group.shape}, index_unique={group.index.is_unique}")
                
                # Step 7: Additional debug - check for NaN values
                nan_counts = group[feature_cols + [target_col]].isna().sum()
                if nan_counts.sum() > 0:
                    logger.warning(f"NaN values found in {ticker}: {nan_counts.to_dict()}")
                
                # Step 8: Check data types
                logger.debug(f"Data types for {ticker}: {group[feature_cols + [target_col]].dtypes.to_dict()}")
                # ==============================================================================

                # Now, proceed with the cleaned group
                X_sequences, y_sequences = _create_scaled_sequences_from_group(group, feature_cols, target_col, window_size)
                
                if X_sequences:
                    all_X[ticker] = X_sequences
                    all_y_ticker[ticker] = y_sequences
                else:
                    logger.warning(f"No sequences generated for ticker {ticker}")
            
            if not all_X:
                logger.error("No sequences were generated for any ticker. Check data and window size.")
                return False, None

            # Now build the training set based on the model type
            if model == 'LSTM_Shuffle':
                logger.info("--- Training with Stateless LSTM (Shuffle) using expanding scaler ---")
                
                # Combine sequences from all tickers
                X_list = [item for sublist in all_X.values() for item in sublist]
                y_list = [item for sublist in all_y_ticker.values() for item in sublist]
                
                X_train, y_train = np.array(X_list), np.array(y_list)
                indices = np.arange(X_train.shape[0])
                np.random.shuffle(indices)
                X_train, y_train = X_train[indices], y_train[indices]
                
                lstm_model = LSTMModel(input_shape=(window_size, len(feature_cols)))
                lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
                lstm_model.save(model_save_path)
                return True, model_save_path

            elif model == 'LSTM_NoShuffle':
                logger.info("--- Training with Stateful LSTM (NoShuffle) using expanding scaler ---")
                
                batch_input_shape = (batch_size, window_size, len(feature_cols))
                lstm_model = LSTMModel(batch_input_shape=batch_input_shape, stateful=True)
                tickers = list(all_X.keys())

                for epoch in range(epochs):
                    np.random.shuffle(tickers)
                    for ticker in tickers:
                        X_train_stock = np.array(all_X[ticker])
                        y_train_stock = np.array(all_y_ticker[ticker])
                        
                        n_samples = len(X_train_stock)
                        trimmed_len = (n_samples // batch_size) * batch_size
                        if trimmed_len == 0: continue
                        
                        X_train_stock = X_train_stock[:trimmed_len]
                        y_train_stock = y_train_stock[:trimmed_len]

                        lstm_model.fit(
                            X_train_stock, y_train_stock, epochs=1,
                            batch_size=batch_size, verbose=0, shuffle=False
                        )
                        lstm_model.reset_states()
                
                lstm_model.save(model_save_path)
                return True, model_save_path
        
        else:
            logger.error(f"Unknown model training type specified: '{model}'")
            return False, None

    except Exception as e:
        logger.error(f"Error during model training execution: {e}", exc_info=True)
        return False, None