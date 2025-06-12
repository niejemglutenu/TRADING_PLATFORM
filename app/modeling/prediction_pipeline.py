# trading_platform/app/modeling/prediction_pipeline.py
import logging
import pandas as pd
import numpy as np
import joblib
import keras # Or from tensorflow import keras
from typing import Tuple, Optional, Any, List

from app.data_ingestion.db_manager import DataManager # Import your DataManager
from app.feature_engineering.strategies import FeatureEngineeringStrategy
# Example for isinstance check, or rely on feature_engineer.get_feature_names()
from app.feature_engineering.strategies import ReturnsVarCorrStrategy 
from app.modeling.iterative_predictor import predict_future_returns_iterative_generalized # Your generalized one
from app.common.constants import FEATURE_CORRELATION, FEATURE_CLOSE # and others as needed

logger = logging.getLogger("app.modeling.prediction_pipeline")

def prediction_pipeline(
    ticker_to_predict: str,
    index_ticker_symbol: Optional[str], # For fetching index data if strategy needs it
    historical_data_fetch_start_date_str: str,
    historical_data_end_date_str: str, # Last day of KNOWN actual data (T-1)
    n_days_to_predict: int,
    model_load_path: str, # Assume paths are always valid strings now
    x_scaler_load_path: str,
    y_scaler_load_path: str,
    feature_engineer: FeatureEngineeringStrategy,
    data_manager: DataManager, # Pass the DataManager instance
    ohlcv_table_name: str,
    output_suffix: str = "" # Optional for logging or unique file names if created here
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:

    logger.info(f"--- Prediction Pipeline for {ticker_to_predict} using Strategy: {feature_engineer.__class__.__name__} ---")
    logger.info(f"Historical data ref end: {historical_data_end_date_str}, Predicting {n_days_to_predict} days")

    try:
        model = keras.models.load_model(model_load_path)
        x_scaler = joblib.load(x_scaler_load_path)
        y_scaler = joblib.load(y_scaler_load_path)
    except Exception as e:
        logger.error(f"Error loading model/scalers for {ticker_to_predict} from {model_load_path}: {e}", exc_info=True)
        return None, None

    # --- Fetch RAW Historical Stock Data (up to T-1) ---
    # This data is for:
    #   a) Generating the initial LSTM feature sequence (via feature_engineer).
    #   b) Providing the historical base for the iterative predictor.
    # `historical_data_fetch_start_date_str` should account for FE strategy's max lookback + LSTM window.
    stock_data_map = data_manager.get_data_from_db(
        tickers_list=[ticker_to_predict],
        start_date_str=historical_data_fetch_start_date_str,
        end_date_str=historical_data_end_date_str, # Fetch up to last known day
        table_name=ohlcv_table_name
    )
    if not stock_data_map or ticker_to_predict not in stock_data_map or stock_data_map[ticker_to_predict].empty:
        logger.error(f"No historical RAW stock data for {ticker_to_predict} from DB. Aborting.")
        return None, None
    historical_stock_raw_df = stock_data_map[ticker_to_predict] # This is df_stock_raw up to T-1

    # --- Fetch RAW Index Data (Historical up to T-1 for initial features, AND Future for iterative predictor) ---
    raw_index_df_for_initial_fe = None # For feature_engineer.generate_features on historical data
    raw_index_df_hist_and_future_for_iterative = None # For iterative_predictor

    # Check if the strategy declares it needs an index (more robust than isinstance)
    # For now, we assume if FEATURE_CORRELATION is among its features, it needs index.
    # A better way: strategy_needs_index = feature_engineer.does_require_index_data() # a new method in base
    strategy_needs_index = FEATURE_CORRELATION in feature_engineer.get_feature_names()

    if strategy_needs_index and index_ticker_symbol:
        try:
            hist_end_dt_utc = pd.to_datetime(historical_data_end_date_str, utc=True)
        except Exception as e:
            logger.error(f"Error parsing historical_data_end_date_str '{historical_data_end_date_str}': {e}"); return None, None

        # Determine end date for fetching index data (needs to cover future predictions)
        index_fetch_end_dt_for_iterative = hist_end_dt_utc
        # Add buffer for BDays; n_days_to_predict is usually small, so a simple loop is fine.
        # A more precise way would use pd.offsets.BDay(n_days_to_predict) and add more buffer.
        for _ in range(n_days_to_predict + 15): # Add ample buffer for weekends/holidays
            index_fetch_end_dt_for_iterative += pd.offsets.BDay(1) # This is not efficient for large n, but ok for small prediction horizons
        
        index_fetch_end_date_str_iterative = index_fetch_end_dt_for_iterative.strftime('%Y-%m-%d')

        logger.info(f"Fetching index data ({index_ticker_symbol}) for full context: "
                     f"{historical_data_fetch_start_date_str} to {index_fetch_end_date_str_iterative}")
        
        index_data_map_full_hist_future = data_manager.get_data_from_db(
            tickers_list=[index_ticker_symbol],
            start_date_str=historical_data_fetch_start_date_str, # Same start as stock raw context
            end_date_str=index_fetch_end_date_str_iterative,    # Up to future date
            table_name=ohlcv_table_name
        )
        if index_data_map_full_hist_future and index_ticker_symbol in index_data_map_full_hist_future and \
           not index_data_map_full_hist_future[index_ticker_symbol].empty:
            
            raw_index_df_hist_and_future_for_iterative = index_data_map_full_hist_future[index_ticker_symbol]
            
            # Slice for initial feature engineering (only up to historical_data_end_date_str)
            raw_index_df_for_initial_fe = raw_index_df_hist_and_future_for_iterative.loc[
                raw_index_df_hist_and_future_for_iterative.index <= hist_end_dt_utc
            ].copy()
            if raw_index_df_for_initial_fe.empty:
                 logging.warning(f"Historical part of index data for {index_ticker_symbol} (up to {hist_end_dt_utc.date()}) is empty.")
                 raw_index_df_for_initial_fe = None # Ensure it's None if slice is empty
        else:
            logging.warning(f"Could not fetch or index data for {index_ticker_symbol} is empty for the required full range.")
    elif strategy_needs_index and not index_ticker_symbol:
        logging.warning("Strategy indicates it needs index data, but no 'index_ticker_symbol' provided.")


    # --- Generate Initial Feature Sequence (from historical data up to T-1) ---
    # feature_engineer.generate_features will use historical_stock_raw_df and raw_index_df_for_initial_fe
    stock_hist_features_transformed_df = feature_engineer.generate_features(
        df_stock_raw=historical_stock_raw_df.copy(), # Pass copy
        df_index_raw=raw_index_df_for_initial_fe.copy() if raw_index_df_for_initial_fe is not None else None
    )
    
    lstm_window_size = feature_engineer.config.get('lstm_window_size', 10)
    ordered_feature_names = feature_engineer.get_feature_names()

    if not ordered_feature_names: # Should be caught by strategy if it doesn't set them
        logging.error("Feature names not set by strategy instance. Aborting."); return None, None
    if stock_hist_features_transformed_df.empty or len(stock_hist_features_transformed_df) < lstm_window_size:
        logging.error(f"Not enough transformed historical features for {ticker_to_predict} ({len(stock_hist_features_transformed_df)} rows) "
                      f"for LSTM window {lstm_window_size}.")
        return None, stock_hist_features_transformed_df # Return for inspection
    if not all(col in stock_hist_features_transformed_df.columns for col in ordered_feature_names):
        missing_cols = [col for col in ordered_feature_names if col not in stock_hist_features_transformed_df.columns]
        logging.error(f"Transformed historical data missing expected features: {missing_cols}. "
                      f"Available: {stock_hist_features_transformed_df.columns.tolist()}. Aborting.")
        return None, stock_hist_features_transformed_df

    # --- Prepare Inputs for Generalized Iterative Predictor ---
    initial_feature_sequence_unscaled_np = stock_hist_features_transformed_df[ordered_feature_names].tail(lstm_window_size).values
    
    # --- Perform Iterative Prediction ---
    predicted_returns_list = predict_future_returns_iterative_generalized(
        model=model,
        initial_feature_sequence_unscaled_np=initial_feature_sequence_unscaled_np,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        n_days_to_predict=n_days_to_predict,
        feature_engineer=feature_engineer, # Pass the strategy instance
        historical_stock_raw_df=historical_stock_raw_df.copy(), # Base raw data for iterative updates
        historical_and_future_index_raw_df=raw_index_df_hist_and_future_for_iterative.copy() if raw_index_df_hist_and_future_for_iterative is not None else None
    )
    
    if not predicted_returns_list: # Check if list is empty
        logging.error(f"Iterative predictor returned no predictions for {ticker_to_predict}.")
        return None, stock_hist_features_transformed_df

    # --- Format and Return Predictions ---
    try:
        forecast_origin_date_utc = pd.to_datetime(historical_data_end_date_str, utc=True)
        last_hist_dt_for_pred_dates = forecast_origin_date_utc
    except Exception as e:
        logger.error(f"Error parsing historical_data_end_date_str '{historical_data_end_date_str}': {e}", exc_info=True)

        # Use the actual last date from the fetched historical stock data index if available and valid
        # otherwise, fallback to the provided historical_data_end_date_str.
        if not historical_stock_raw_df.empty and isinstance(historical_stock_raw_df.index, pd.DatetimeIndex):
            last_hist_dt_for_pred_dates = historical_stock_raw_df.index[-1]
        else:
            last_hist_dt_for_pred_dates = pd.Timestamp.now(tz='UTC').normalize() - pd.offsets.BDay(1)

    tz_info_pred_dates = last_hist_dt_for_pred_dates.tz # Ensure timezone from source
    start_for_bdate_range = pd.Timestamp(last_hist_dt_for_pred_dates) + pd.offsets.BDay(1)

    generated_prediction_dates = pd.bdate_range(
        start=start_for_bdate_range,
        periods=n_days_to_predict, # Ensure enough dates are generated
        tz=tz_info_pred_dates
        )


    if len(generated_prediction_dates) < len(predicted_returns_list):
        logger.warning(f"Generated fewer prediction dates ({len(generated_prediction_dates)}) "
                       f"than predictions ({len(predicted_returns_list)}). Truncating predictions.")
        predicted_returns_list = predicted_returns_list[:len(generated_prediction_dates)]
    
    predictions_df = pd.DataFrame({
        'Ticker': ticker_to_predict,
        'ForecastOriginDate': forecast_origin_date_utc, # Day T-1 (when forecast was made)
        'PredictionDate': generated_prediction_dates[:len(predicted_returns_list)], # Dates T, T+1, ...
        'PredictedReturn': predicted_returns_list
    })

    # Calculate ForecastHorizon in business days
    # This requires a common index or careful application if dates are not perfectly aligned
    # A robust way is to count business days between ForecastOriginDate and PredictionDate
    horizons = []
    if not predictions_df.empty:
        for _, row in predictions_df.iterrows():
            # Count business days from the day *after* ForecastOriginDate up to PredictionDate
            horizon = len(pd.bdate_range(start=row['ForecastOriginDate'] + pd.offsets.BDay(1), 
                                         end=row['PredictionDate'], 
                                         tz=row['PredictionDate'].tz))
            horizons.append(horizon)
        predictions_df['ForecastHorizon'] = horizons
    else:
        predictions_df['ForecastHorizon'] = []


    logger.info(f"Predicted Returns for {ticker_to_predict} (suffix: {output_suffix}):\n{predictions_df.to_string(index=False)}")
    
    return predictions_df, stock_hist_features_transformed_df