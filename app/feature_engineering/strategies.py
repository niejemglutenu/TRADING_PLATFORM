# trading_platform/app/feature_engineering/strategies.py
import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Type, Any
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)

class FeatureEngineeringStrategy(ABC):
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._target_col_name = "returns"

    def get_required_raw_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close', 'volume']

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Returns a definitive, static list of feature names."""
        pass

    @abstractmethod
    def _calculate_features(self, df: pd.DataFrame, df_index_raw: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calculates and adds features to the DataFrame."""
        pass

    @abstractmethod
    def generate_synthetic_row(self, last_known_row: pd.Series, predicted_value: float) -> pd.DataFrame:
        """Creates a synthetic future data row from a prediction."""
        pass

    def generate_features(self, df_stock_raw: pd.DataFrame, df_index_raw: Optional[pd.DataFrame] = None, prediction_horizon: int = 1) -> pd.DataFrame:
        """Public method to orchestrate feature generation and target shifting."""
        if df_stock_raw.empty:
            return pd.DataFrame()
        
        df = df_stock_raw.copy()
        df['returns'] = df['close'].pct_change()
        
        df_with_features = self._calculate_features(df, df_index_raw)
        
        target_values = df_with_features[self._target_col_name].shift(-prediction_horizon)
        df_with_features['target_returns'] = target_values
        
        return df_with_features

    def get_target_name(self) -> str:
        return 'target_returns'  # Use a distinct name that won't conflict with features

    def requires_index_data(self) -> bool:
        return False

    @staticmethod
    def create(strategy_key: str, config: Optional[Dict] = None) -> 'FeatureEngineeringStrategy':
        # This map needs to be kept up-to-date
        strategy_map: Dict[str, Type[FeatureEngineeringStrategy]] = {
            'PastReturnsStrategy': PastReturnsStrategy,
            'ReturnsVariationStrategy': ReturnsVariationStrategy,
            'ReturnsRelativeStrengthStrategy': ReturnsRelativeStrengthStrategy,
            'TechnicalIndicatorsForReturn': TechnicalIndicatorsForReturn,
            'TechnicalIndicatorsForSharpe': TechnicalIndicatorsForSharpe

        }
        strategy_class = strategy_map.get(strategy_key)
        if strategy_class is None:
            raise ValueError(f"Unknown feature engineering strategy: {strategy_key}")
        return strategy_class(config=config)



class PastReturnsStrategy(FeatureEngineeringStrategy):
    def get_feature_names(self) -> List[str]:
        return ['returns']

    def _calculate_features(self, df: pd.DataFrame, df_index_raw: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        # The 'returns' column is already calculated in the base generate_features.
        # We just need to handle the initial NaN.
        df['returns'] = df['returns'].fillna(0)
        return df

    def generate_synthetic_row(self, last_known_row: pd.Series, predicted_value: float) -> pd.DataFrame:
        if np.isnan(predicted_value):
            logger.error("generate_synthetic_row received a NaN predicted_value! Using 0 as fallback.")
            predicted_value = 0

        last_close = last_known_row['close']
        new_close = last_close * (1 + predicted_value)
        new_date = last_known_row.name + pd.offsets.BDay(1)
        # We must return all columns needed for the next feature calculation
        return pd.DataFrame({
            'open': new_close, 'high': new_close, 'low': new_close, 'close': new_close,
            'volume': last_known_row.get('volume', 0) # Use .get for safety
        }, index=[new_date])







class ReturnsVariationStrategy(PastReturnsStrategy):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.variation_window = self.config.get('variation_window', 10)
        self.min_periods_variation = self.config.get('min_periods_variation', 5)
        
    def get_feature_names(self) -> List[str]:
        return super().get_feature_names() + ['returns_std']

    def _calculate_features(self, df: pd.DataFrame, df_index_raw: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        df_out = super()._calculate_features(df, df_index_raw)
        df_out['returns_std'] = df_out['returns'].rolling(
            window=self.variation_window, 
            min_periods=self.min_periods_variation
        ).std().fillna(0) # Fill NaNs from rolling operation
        return df_out

class ReturnsRelativeStrengthStrategy(ReturnsVariationStrategy):
    def requires_index_data(self) -> bool:
        return True
        
    def get_feature_names(self) -> List[str]:
        return super().get_feature_names() + ['relative_return']

    def _calculate_features(self, df: pd.DataFrame, df_index_raw: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        df_out = super()._calculate_features(df, df_index_raw)
        
        if df_index_raw is not None and not df_index_raw.empty:
            index_returns = df_index_raw['close'].pct_change().rename('index_returns')
            df_out_clean = df_out.copy()
            index_returns_clean = index_returns.copy()
            if df_out_clean.index.has_duplicates:
                df_out_clean = df_out_clean.loc[~df_out_clean.index.duplicated(keep='first')]
            if index_returns_clean.index.has_duplicates:
                index_returns_clean = index_returns_clean.loc[~index_returns_clean.index.duplicated(keep='first')]
            merged_df = pd.merge(
                df_out_clean, 
                index_returns_clean, 
                left_index=True, 
                right_index=True, 
                how='left'
            )
            if merged_df.index.has_duplicates:
                logger.warning("Duplicate timestamps detected after merge in ReturnsRelativeStrengthStrategy")
                merged_df = merged_df.loc[~merged_df.index.duplicated(keep='first')]
            cleaned_index_returns = merged_df['index_returns'].ffill().fillna(0)
            merged_df['relative_return'] = merged_df['returns'] - cleaned_index_returns
            df_out = merged_df.drop(columns=['index_returns'])
        else:
            logger.warning("Index data not provided for Relative Strength. Setting 'relative_return' to 0.")
            df_out['relative_return'] = 0.0
        return df_out
    

import backtrader as bt

class BaseIndicatorStrategy(FeatureEngineeringStrategy):
    """
    A base class that calculates a rich set of technical indicators
    using the built-in Backtrader indicator library.
    """
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.rsi_period = self.config.get('rsi_period', 14)
        self.macd_fast = self.config.get('macd_fast', 12)
        self.macd_slow = self.config.get('macd_slow', 26)
        self.macd_signal = self.config.get('macd_signal', 9)
        self.sma_fast_period = self.config.get('sma_fast_period', 20)
        self.sma_slow_period = self.config.get('sma_slow_period', 50)
    
    def get_required_raw_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close', 'volume']

    def _calculate_features(self, df: pd.DataFrame, df_index_raw: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculates all technical indicators using Backtrader and sets them as features.
        """
        if df.empty:
            return pd.DataFrame()
        

        df_for_bt = df.copy()
        df_for_bt.columns = [x.lower() for x in df_for_bt.columns]
        data_feed = bt.feeds.PandasData(dataname=df_for_bt)

        
        # 1. RSI
        rsi = bt.indicators.RSI(data_feed.close, period=self.rsi_period)
        df[f'RSI_{self.rsi_period}'] = rsi.array

        # 2. MACD
        # The MACD indicator in backtrader returns multiple lines
        macd = bt.indicators.MACD(
            data_feed.close,
            period_me1=self.macd_fast,
            period_me2=self.macd_slow,
            period_signal=self.macd_signal
        )
        df[f'MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'] = macd.macd.array
        df[f'MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'] = macd.histo.array
        df[f'MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'] = macd.signal.array

        # 3. SMAs
        sma_fast = bt.indicators.SimpleMovingAverage(data_feed.close, period=self.sma_fast_period)
        sma_slow = bt.indicators.SimpleMovingAverage(data_feed.close, period=self.sma_slow_period)
        df[f'SMA_{self.sma_fast_period}'] = sma_fast.array
        df[f'SMA_{self.sma_slow_period}'] = sma_slow.array

        # 4. Custom Feature: SMA Ratio
        df['sma_ratio'] = df[f'SMA_{self.sma_fast_period}'] / df[f'SMA_{self.sma_slow_period}']
        
        # 5. Log Returns
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Define the list of feature names we just created
        self._feature_names = [
            f'RSI_{self.rsi_period}',
            f'MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}',
            f'MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}',
            f'MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}',
            'sma_ratio',
            'log_returns'
        ]
        
        # Drop NaNs created by the indicators at the beginning of the series
        return df.dropna(subset=self._feature_names)

    # The generate_synthetic_row method does not need to change.
    # It works perfectly with this new implementation.
    def generate_synthetic_row(self, last_known_row: pd.Series, predicted_value: float) -> pd.DataFrame:
        last_close = last_known_row['close']
        
        if "sharpe" in self._target_col_name.lower():
            predicted_return = 0.001 * predicted_value
        else:
            predicted_return = predicted_value

        new_close = last_close * (1 + predicted_return)
        new_date = last_known_row.name + pd.offsets.BDay(1)
        
        return pd.DataFrame({
            'open': new_close, 'high': new_close, 'low': new_close, 'close': new_close, 'volume': last_known_row['volume']
        }, index=[new_date])

# =========================================================================
# STRATEGY 1: Use technical indicators to predict next-day return
# =========================================================================
class TechnicalIndicatorsForReturn(BaseIndicatorStrategy):
    """
    This strategy calculates a rich set of features and uses them
    to predict the standard next-day return.
    """
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        # The target is the standard 'returns' column.
        self._target_col_name = "returns"

# =========================================================================
# STRATEGY 2: Use technical indicators to predict future Sharpe Ratio
# =========================================================================
class TechnicalIndicatorsForSharpe(BaseIndicatorStrategy):
    """
    This strategy calculates a rich set of features and uses them
    to predict the rolling Sharpe Ratio over a future period.
    """
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.sharpe_period = self.config.get('sharpe_period', 20) # e.g., next 20 days
        
        # The target is a new, custom column we will create.
        self._target_col_name = f"future_sharpe_{self.sharpe_period}"

    # We need to override the main generation method to create our custom target
    def generate_features(self, df_stock_raw: pd.DataFrame, df_index_raw: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Orchestrates feature generation and creates the custom future Sharpe Ratio target.
        """
        if df_stock_raw.empty:
            return pd.DataFrame()
        
        df_with_features = df_stock_raw.copy()
        
        # --- 1. Calculate the base return needed for the target ---
        df_with_features['returns'] = df_with_features['close'].pct_change()
        
        # --- 2. Create the custom target: Future Sharpe Ratio ---
        # Calculate the rolling mean of future returns
        future_returns_mean = df_with_features['returns'].rolling(window=self.sharpe_period).mean().shift(-self.sharpe_period)
        # Calculate the rolling std dev of future returns
        future_returns_std = df_with_features['returns'].rolling(window=self.sharpe_period).std().shift(-self.sharpe_period)
        
        # Calculate Sharpe, replacing division by zero with 0
        df_with_features[self.get_target_name()] = (future_returns_mean / future_returns_std).fillna(0)
        # Annualize the Sharpe Ratio (optional, but standard practice)
        df_with_features[self.get_target_name()] *= np.sqrt(252)

        # --- 3. Let the parent class calculate all the input features (X) ---
        df_with_features = self._calculate_features(df_with_features, df_index_raw)
        
        # NOTE: We do NOT call shift(-1) here because we already created our future-looking target.
        return df_with_features