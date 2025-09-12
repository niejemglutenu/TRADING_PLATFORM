import logging
import pandas as pd
from typing import Tuple, Optional

from app.data_ingestion.db_manager import DataManager
from app.feature_engineering.strategies import FeatureEngineeringStrategy
from app.modeling.model_builders import LSTMModel
from app.modeling.iterative_predictor import predict_future_iterative

logger = logging.getLogger("app.modeling.prediction_pipeline")

############### actal raw data ---> features ----->  
def prediction_pipeline(
    ticker_to_predict: str,
    index_ticker_symbol: Optional[str],
    historical_data_fetch_start_date_str: str,
    historical_data_end_date_str: str,
    n_days_to_predict: int,
    model_load_path: str,
    feature_engineer: FeatureEngineeringStrategy,
    data_manager: DataManager,
    ohlcv_table_name: str
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    try:
        logger.info(f"--- Prediction Pipeline Started for Ticker: {ticker_to_predict}, Horizon: {n_days_to_predict} days ---")

        model = LSTMModel()
        if not model.load(model_load_path):
            return None, None

        hist_start_dt = pd.to_datetime(historical_data_fetch_start_date_str)
        hist_end_dt = pd.to_datetime(historical_data_end_date_str)
        # Calculate the end date for our future data fetch, add a buffer
        future_end_dt = hist_end_dt + pd.Timedelta(days=n_days_to_predict + 5)

        stock_data_map = data_manager.get_data_from_db(
            [ticker_to_predict], # Pass as a list
            hist_start_dt.strftime('%Y-%m-%d'),
            hist_end_dt.strftime('%Y-%m-%d'),
            ohlcv_table_name
        )
        stock_df_raw = stock_data_map.get(ticker_to_predict)

        if stock_df_raw is None or stock_df_raw.empty:
            logger.error(f"No historical data found for {ticker_to_predict} in the given date range.")
            return None, None
            
        index_df_full = None
        if feature_engineer.requires_index_data():
            if not index_ticker_symbol:
                logger.error("Feature strategy requires an index, but no index_ticker_symbol was provided.")
                return None, None
            
            logger.info(f"Fetching full index data ({index_ticker_symbol}) from {hist_start_dt.date()} to {future_end_dt.date()}")
            
            index_data_map = data_manager.get_data_from_db(
                [index_ticker_symbol], 
                hist_start_dt.strftime('%Y-%m-%d'),
                future_end_dt.strftime('%Y-%m-%d'), 
                ohlcv_table_name
            )
            index_df_full = index_data_map.get(index_ticker_symbol)

            if index_df_full is None or index_df_full.empty:
                logger.error(f"Could not fetch required index data for {index_ticker_symbol}.")
                return None, None
        
        feature_df = feature_engineer.generate_features(stock_df_raw, index_df_full)
        feature_cols = feature_engineer.get_feature_names()
        target_col = feature_engineer.get_target_name() 
        feature_df.dropna(subset=feature_cols, inplace=True)
        if feature_df.empty: return None, None

        window_size = model.model.input_shape[1]
        
        cols_for_scaling = feature_cols + [target_col]
        
        df_for_scaling = feature_df[cols_for_scaling].copy()
        
        expanding_mean = df_for_scaling.expanding(min_periods=window_size).mean()
        expanding_std = df_for_scaling.expanding(min_periods=window_size).std()
        
        scaled_feature_df = pd.DataFrame(index=df_for_scaling.index)
        for col in feature_cols:
            scaled_feature_df[col] = (df_for_scaling[col] - expanding_mean[col]) / (expanding_std[col] + 1e-6)

        initial_sequence_scaled = scaled_feature_df.tail(window_size).values
        if len(initial_sequence_scaled) < window_size: return None, None
            
        last_target_mean = expanding_mean[target_col].iloc[-1]
        last_target_std = expanding_std[target_col].iloc[-1]

        predictions_df = predict_future_iterative(
            model=model,
            initial_sequence_scaled=initial_sequence_scaled,
            last_target_mean=last_target_mean,
            last_target_std=last_target_std,
            n_steps_to_predict=n_days_to_predict,
            feature_engineer=feature_engineer,
            historical_stock_raw_df=stock_df_raw,
            full_index_raw_df=index_df_full 
        )

        if predictions_df is None or predictions_df.empty:
            logger.error(f"Iterative prediction failed for {ticker_to_predict}.")
            return None, None
            
        predictions_df['Ticker'] = ticker_to_predict
        logger.info(f"--- Prediction Pipeline Finished for {ticker_to_predict} ---")
        return predictions_df, feature_df

    except Exception as e:
        logger.error(f"Error in prediction pipeline for {ticker_to_predict}: {e}", exc_info=True)
        return None, None