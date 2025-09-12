from typing import Tuple, List, Optional
from keras import backend
from keras.models import Sequential, Model
from keras.layers import LSTM, Bidirectional, LayerNormalization, Dropout, Dense, LeakyReLU, Input
from keras import Model
from typing import Any, Dict
import numpy as np
import pandas as pd
import logging
from app.common.config import AppConfig
import tensorflow as tf
import joblib
import json
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def create_lstm_model(input_shape: Tuple[int, int]) -> Model:
    inputs = Input(shape=input_shape)
    x = LSTM(50, return_sequences=True)(inputs)
    x = LSTM(50)(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


def create_stateful_lstm_model(batch_input_shape: Tuple[Optional[int], int, int]) -> Model:
    inputs = Input(batch_shape=batch_input_shape)
    x = LSTM(50, return_sequences=True, stateful=True)(inputs)
    x = LSTM(50, stateful=True)(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    return model


class LSTMModel:
    def __init__(self, input_shape: Optional[Tuple[int, int]] = None, batch_input_shape: Optional[Tuple[int, int, int]] = None, stateful: bool = False):
        self.input_shape = input_shape
        self.batch_input_shape = batch_input_shape
        self.stateful = stateful
        self.model: Optional[Model] = None
        self.x_scaler: Optional[StandardScaler] = None
        self.y_scaler: Optional[StandardScaler] = None
        
        if self.stateful and self.batch_input_shape:
            self.model = create_stateful_lstm_model(self.batch_input_shape)
        elif not self.stateful and self.input_shape:
            self.model = create_lstm_model(self.input_shape)

    def fit(self, X, y, **kwargs):
        if self.model is None:
             if self.stateful:
                 self.batch_input_shape = (kwargs.get('batch_size'), X.shape[1], X.shape[2])
                 self.model = create_stateful_lstm_model(self.batch_input_shape)
             else:
                 self.input_shape = (X.shape[1], X.shape[2])
                 self.model = create_lstm_model(self.input_shape)
        self.model.fit(X, y, **kwargs)




    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not initialized. Call fit() or load() a trained model.")
        return self.model.predict(X)

    def prepare_training_data(self, data_df: pd.DataFrame, feature_cols: List[str], target_col: str, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        from app.feature_engineering.feature_utils import create_X_Y_sequenced_for_training
        X, y = create_X_Y_sequenced_for_training(data_df, feature_cols, target_col, window_size)
        return X, y

    def save(self, path: str):
        try:
            model_dir = Path(path)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            self.model.save(model_dir / "model.keras")

            if self.x_scaler is not None:
                joblib.dump(self.x_scaler, model_dir / "x_scaler.joblib")
            
            if self.y_scaler is not None:
                joblib.dump(self.y_scaler, model_dir / "y_scaler.joblib")
            
            logger.info(f"Model and scalers saved successfully to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}", exc_info=True)
            return False

    def load(self, path: str) -> bool:
        try:
            model_dir = Path(path)
            
            self.model = tf.keras.models.load_model(model_dir / "model.keras")
            
            if self.model and self.model.input_shape:
                 self.input_shape = self.model.input_shape[1:]
            
            x_scaler_path = model_dir / "x_scaler.joblib"
            if x_scaler_path.exists():
                self.x_scaler = joblib.load(x_scaler_path)
            
            y_scaler_path = model_dir / "y_scaler.joblib"
            if y_scaler_path.exists():
                self.y_scaler = joblib.load(y_scaler_path)
            
            logger.info(f"Model and scalers loaded successfully from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}", exc_info=True)
            return False
        

    def reset_states(self):
        if self.model is None:
            logger.warning("Attempted to reset states on a model that has not been built yet.")
            return

        if self.stateful:
            for layer in self.model.layers:
                if hasattr(layer, 'stateful') and layer.stateful:
                    layer.reset_states()
                    logger.debug(f"Resetting states for stateful layer: {layer.name}")
        else:
            logger.debug("Model is stateless, no states to reset.")
            pass