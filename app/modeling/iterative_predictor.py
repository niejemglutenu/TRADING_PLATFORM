import logging
import pandas as pd
import numpy as np
from typing import List, Optional
import tensorflow as tf

from app.feature_engineering.strategies import FeatureEngineeringStrategy
from app.modeling.model_builders import LSTMModel

logger = logging.getLogger(__name__)

def predict_future_iterative(
    model: LSTMModel,
    initial_sequence_scaled: np.ndarray,
    last_target_mean: float,
    last_target_std: float,
    n_steps_to_predict: int,
    feature_engineer: FeatureEngineeringStrategy,
    historical_stock_raw_df: pd.DataFrame,
    full_index_raw_df: Optional[pd.DataFrame] = None
) -> Optional[pd.DataFrame]:
    try:
        if model.model is None:
            logger.error("Iterative Predictor: Model is not loaded.")
            return None

        window_size = model.model.input_shape[1]
        feature_names = feature_engineer.get_feature_names()

        current_sequence_scaled = initial_sequence_scaled
        df_with_synthetic = historical_stock_raw_df.copy()
        predictions = []
        
        current_target_mean = last_target_mean
        current_target_std = last_target_std
        
        for day in range(n_steps_to_predict):
            input_for_pred = current_sequence_scaled.reshape(1, window_size, len(feature_names))
            
            scaled_prediction = model.model.predict(input_for_pred, verbose=0)
            
            predicted_return = (scaled_prediction[0][0] * current_target_std) + current_target_mean

            last_known_date = df_with_synthetic.index[-1]
            prediction_target_date = last_known_date + pd.offsets.BDay(1)

            predictions.append({
                'PredictedValue': predicted_return,
                'PredictionDate': prediction_target_date,
                'ForecastHorizon': day + 1,
                'ForecastOriginDate': historical_stock_raw_df.index[-1]
            })
            synthetic_row = feature_engineer.generate_synthetic_row(df_with_synthetic.iloc[-1], predicted_return)
            df_with_synthetic = pd.concat([df_with_synthetic, synthetic_row])

            extended_features_df = feature_engineer.generate_features(
                df_with_synthetic, df_index_raw=full_index_raw_df
            )
            
            cols_for_scaling = feature_names + [feature_engineer.get_target_name()]
            new_expanding_mean = extended_features_df[cols_for_scaling].expanding(min_periods=window_size).mean()
            new_expanding_std = extended_features_df[cols_for_scaling].expanding(min_periods=window_size).std()

            scaled_feature_df = pd.DataFrame(index=extended_features_df.index)
            for col in feature_names:
                scaled_feature_df[col] = (extended_features_df[col] - new_expanding_mean[col]) / (new_expanding_std[col] + 1e-6)

            current_sequence_scaled = scaled_feature_df.tail(window_size).values

            current_target_mean = new_expanding_mean[feature_engineer.get_target_name()].iloc[-1]
            current_target_std = new_expanding_std[feature_engineer.get_target_name()].iloc[-1]

        final_predictions = pd.DataFrame(predictions)
        final_predictions.rename(columns={'PredictedValue': 'PredictedReturn'}, inplace=True)
        return final_predictions

    except Exception as e:
        logger.error(f"Error during iterative prediction: {e}", exc_info=True)
        return None