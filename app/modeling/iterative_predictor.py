# trading_platform/app/modeling/iterative_predictor.py
import logging
import pandas as pd
import numpy as np
from typing import List, Optional, Any

# Assuming FeatureEngineeringStrategy and specific strategies are importable
# For type hinting and isinstance checks if needed.
from app.feature_engineering.strategies import FeatureEngineeringStrategy
# from app.feature_engineering.strategies.advanced_features import ReturnsVarCorrStrategy # Example for isinstance
from app.common.constants import FEATURE_CLOSE # Assuming 'close' is your standard raw close column name

logger = logging.getLogger("app.modeling.iterative_predictor")

def predict_future_returns_iterative_generalized(
    model: Any, # Keras model
    initial_feature_sequence_unscaled_np: np.ndarray, # Shape: (lstm_window_size, num_features)
    x_scaler: Any, # Fitted StandardScaler for X
    y_scaler: Any, # Fitted StandardScaler for Y
    n_days_to_predict: int,
    feature_engineer: FeatureEngineeringStrategy, # The instantiated strategy object
    historical_stock_raw_df: pd.DataFrame, # Raw stock data up to day T-1 (DatetimeIndex, must have FEATURE_CLOSE)
    historical_and_future_index_raw_df: Optional[pd.DataFrame] = None # Raw index data (hist+future, DatetimeIndex)
) -> List[float]:
    """
    Iteratively predicts future returns by:
    1. Predicting next return using the current feature sequence.
    2. Calculating a pseudo 'close' price based on the predicted return.
    3. Appending this pseudo-raw data point (new 'close' on next business day) to the stock's raw data history.
    4. Using the passed 'feature_engineer' to re-calculate ALL features on this updated raw stock history
       (and corresponding index data if the strategy requires it).
    5. Extracting the latest feature vector from the re-calculated features.
    6. Appending this new feature vector to the LSTM input sequence for the next prediction.
    """
    predicted_target_values_list: List[float] = []

    lstm_window_size = feature_engineer.config.get('lstm_window_size', 10)
    ordered_feature_names = feature_engineer.get_feature_names()
    required_raw_cols_for_fe = feature_engineer.get_required_raw_columns() # Usually just [FEATURE_CLOSE]

    if not ordered_feature_names:
        logger.error("Iterative Predictor: No feature names from feature_engineer. Cannot proceed.")
        return []
    num_model_features = len(ordered_feature_names)

    if model.input_shape[-1] != num_model_features:
        logger.error(f"Model input shape mismatch! Model expects {model.input_shape[-1]} features, "
                      f"but strategy '{feature_engineer.__class__.__name__}' produces {num_model_features} "
                      f"features: {ordered_feature_names}")
        return []
    if FEATURE_CLOSE not in historical_stock_raw_df.columns:
        logger.error(f"Iterative Predictor: '{FEATURE_CLOSE}' column missing in historical_stock_raw_df.")
        return []

    current_lstm_input_sequence_np = initial_feature_sequence_unscaled_np.copy()
    current_evolving_stock_raw_df = historical_stock_raw_df.copy() # This DataFrame will grow

    last_known_close = current_evolving_stock_raw_df[FEATURE_CLOSE].iloc[-1]
    last_known_date = current_evolving_stock_raw_df.index[-1]

    logger.debug(f"Iterative predictor starting. Initial sequence shape: {current_lstm_input_sequence_np.shape}. "
                  f"Last known close: {last_known_close:.2f} on {last_known_date.date()}. "
                  f"Predicting for {n_days_to_predict} days. Features: {ordered_feature_names}")

    for day_idx in range(n_days_to_predict):
        if current_lstm_input_sequence_np.shape[0] < lstm_window_size:
            logger.error(f"Iterative Pred Day {day_idx+1}: Not enough feature data rows "
                          f"({current_lstm_input_sequence_np.shape[0]}) for LSTM window {lstm_window_size}.")
            break
        
        lstm_input_unscaled_for_pred = current_lstm_input_sequence_np[-lstm_window_size:, :]
        
        scaled_features_for_lstm = x_scaler.transform(lstm_input_unscaled_for_pred)
        scaled_input_to_model = scaled_features_for_lstm.reshape(1, lstm_window_size, num_model_features)

        scaled_predicted_target = model.predict(scaled_input_to_model, verbose=0)
        predicted_next_day_return = float(y_scaler.inverse_transform(scaled_predicted_target)[0, 0])
        predicted_target_values_list.append(predicted_next_day_return)

        # If it's the last prediction needed, no need to update features for the *next* step
        if day_idx == n_days_to_predict - 1:
            break

        # --- Construct pseudo-raw data for the predicted day and re-calculate features ---
        next_pseudo_close = last_known_close * (1 + predicted_next_day_return)
        next_business_date = last_known_date + pd.offsets.BDay(1)
        
        # Create a new raw data row. Only 'close' is strictly from prediction.
        # Other OHLCV values need a strategy (e.g., set O=H=L=C, carry forward volume, or more complex).
        # The feature_engineer's get_required_raw_columns() tells us what's needed.
        new_raw_row_data = {col: np.nan for col in required_raw_cols_for_fe}
        new_raw_row_data[FEATURE_CLOSE] = next_pseudo_close # Essential
        
        # Simple strategy for other common raw columns if needed by the FE strategy:
        if 'open' in required_raw_cols_for_fe: new_raw_row_data['open'] = next_pseudo_close
        if 'high' in required_raw_cols_for_fe: new_raw_row_data['high'] = next_pseudo_close
        if 'low' in required_raw_cols_for_fe: new_raw_row_data['low'] = next_pseudo_close
        if 'volume' in required_raw_cols_for_fe and 'volume' in current_evolving_stock_raw_df.columns:
            # Use mean of recent volume as a placeholder if strategy needs it
            new_raw_row_data['volume'] = current_evolving_stock_raw_df['volume'].iloc[-20:].mean()
            if pd.isna(new_raw_row_data['volume']): new_raw_row_data['volume'] = 0 # Fallback
        
        new_raw_row_df = pd.DataFrame(new_raw_row_data, index=[next_business_date])
        current_evolving_stock_raw_df = pd.concat([current_evolving_stock_raw_df, new_raw_row_df])

        # Prepare corresponding index data slice for the feature engineer
        index_data_for_this_recalc_step = None
        if historical_and_future_index_raw_df is not None:
            # The strategy will operate on data up to next_business_date
            min_hist_date_for_recalc = current_evolving_stock_raw_df.index.min() # Start of current evolving history
            max_hist_date_for_recalc = next_business_date # End date for current recalc
            
            index_data_for_this_recalc_step = historical_and_future_index_raw_df.loc[
                (historical_and_future_index_raw_df.index >= min_hist_date_for_recalc) &
                (historical_and_future_index_raw_df.index <= max_hist_date_for_recalc)
            ].copy() # Ensure it's a copy
            # It's crucial that historical_and_future_index_raw_df actually HAS data for next_business_date
            # if the feature_engineer requires correlation on that day.

        # Re-generate ALL features using the strategy on the updated raw history
        all_features_recalculated_df = feature_engineer.generate_features(
            df_stock_raw=current_evolving_stock_raw_df, # Pass the full updated raw history
            df_index_raw=index_data_for_this_recalc_step
        )

        if all_features_recalculated_df.empty or len(all_features_recalculated_df) < 1:
            logger.error(f"Iterative Pred Day {day_idx+1}: Feature recalculation on updated raw data resulted in empty DataFrame. Cannot proceed.")
            break
        
        try:
            # Extract the LATEST feature vector (corresponding to next_business_date)
            latest_feature_vector_unscaled_series = all_features_recalculated_df[ordered_feature_names].iloc[-1]
            latest_feature_vector_unscaled_np = np.asarray(latest_feature_vector_unscaled_series.values).reshape(1, num_model_features)
        except (IndexError, KeyError) as e_extract:
            logger.error(f"Iterative Pred Day {day_idx+1}: Error extracting latest feature vector after recalc: {e_extract}. "
                          f"Available columns: {all_features_recalculated_df.columns.tolist()}. "
                          f"Expected: {ordered_feature_names}. DF shape: {all_features_recalculated_df.shape}. "
                          f"Last date in recalc df: {all_features_recalculated_df.index[-1] if not all_features_recalculated_df.empty else 'N/A'}")
            break
        
        current_lstm_input_sequence_np = np.vstack([
            current_lstm_input_sequence_np,
            latest_feature_vector_unscaled_np
        ])

        # Update for next iteration
        last_known_close = next_pseudo_close
        last_known_date = next_business_date

        # Optional trimming of current_evolving_stock_raw_df to prevent excessive memory usage
        # Needs to be long enough for the longest rolling window in feature_engineer.config
        # max_lookback_needed_by_fe = 0 # Calculate this based on feature_engineer.config windows
        # if len(current_evolving_stock_raw_df) > max_lookback_needed_by_fe + lstm_window_size + 20: # Example buffer
        #     current_evolving_stock_raw_df = current_evolving_stock_raw_df.iloc[-(max_lookback_needed_by_fe + lstm_window_size + 10):]

    return predicted_target_values_list