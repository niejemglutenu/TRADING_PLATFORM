import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from abc import ABC, abstractmethod
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, LayerNormalization, LeakyReLU
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import joblib
from app.common.constants import FEATURE_CLOSE, FEATURE_CORRELATION, FEATURE_VARIATION, FEATURE_TARGET, FEATURE_RETURNS


# trading_platform/app/feature_engineering/base_strategy.py
# (Or wherever your FeatureEngineeringStrategy class is defined - likely strategies.py in your case)
import logging
import numpy as np # Should be imported if used by subclasses
import pandas as pd
from typing import List, Optional, Dict
from abc import ABC, abstractmethod
# from app.common.constants import FEATURE_TARGET, FEATURE_CLOSE, FEATURE_RETURNS, FEATURE_VARIATION, FEATURE_CORRELATION
# ^^^ Ensure these constants are defined or imported if base class uses them directly

class FeatureEngineeringStrategy(ABC):
    def __init__(self, config: Optional[Dict] = None ):
        self.config = config if config is not None else {}
        self.feature_names: List[str] = []
        # Default target name (can be overridden by config or subclass)
        self.target_name: str = self.config.get('target_column_name', FEATURE_TARGET) 
        self.lstm_window_size = self.config.get('lstm_window_size', 10)
        # Target shift for predicting further out (e.g., -1 for next period, -2 for period after next)
        self.target_shift_periods = self.config.get('target_shift_periods', -1) 

    @abstractmethod
    def get_required_raw_columns(self) -> List[str]:
        """Returns list of raw column names this strategy needs from stock data (e.g., ['close', 'volume'])."""
        pass

    @abstractmethod
    def _calculate_base_features(self, df_stock_raw: pd.DataFrame, df_index_raw: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Concrete strategies implement this to calculate all their features
        and set self.feature_names. Should return df with all features.
        Target is added by the base class's generate_features.
        """
        pass

    def generate_features(self, df_stock_raw: pd.DataFrame, df_index_raw: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Template method: calculates base features, then adds the shifted target.
        """
        df_with_base_features = self._calculate_base_features(df_stock_raw, df_index_raw)

        if df_with_base_features.empty:
            return pd.DataFrame()

        if FEATURE_RETURNS not in df_with_base_features.columns: # Assuming target is based on returns
            logging.error(f"'{FEATURE_RETURNS}' missing after _calculate_base_features for {self.__class__.__name__}.")
            return pd.DataFrame()
            
        # Adjust target name if shifting differently from default
        if self.target_shift_periods != -1:
            self.target_name = f"{FEATURE_TARGET}_shifted_{abs(self.target_shift_periods)}"
        
        df_with_base_features[self.target_name] = df_with_base_features[FEATURE_RETURNS].shift(self.target_shift_periods).astype(float)

        # Ensure all declared feature_names and the target_name exist before selecting
        final_columns = self.get_feature_names() + [self.get_target_name()]
        missing_cols = [col for col in final_columns if col not in df_with_base_features.columns]
        if missing_cols:
            logging.error(f"Strategy {self.__class__.__name__} generate_features: Missing columns {missing_cols}. Available: {df_with_base_features.columns.tolist()}")
            return pd.DataFrame()

        output_df = df_with_base_features[final_columns].copy()
        return output_df.dropna()
    
    def get_feature_names(self) -> List[str]:
        if not self.feature_names: # Features should be set by _calculate_base_features
            logging.warning(f"Feature names not explicitly set by _calculate_base_features for {self.__class__.__name__}. Call it first or set in __init__.")
        return self.feature_names

    def get_target_name(self) -> str:
        # Target name is now potentially dynamic based on shift, ensure it's current
        if self.target_shift_periods != -1:
            return f"{FEATURE_TARGET}_shifted_{abs(self.target_shift_periods)}"
        return FEATURE_TARGET # Default target name

    def requires_index_data(self) -> bool:
        """
        Concrete strategies should override this to return True if they need df_index_raw.
        Default is False.
        """
        return False

class PastReturnsStrategy(FeatureEngineeringStrategy):
    def get_required_raw_columns(self) -> List[str]: return [FEATURE_CLOSE]

    def _calculate_base_features(self, df_stock_raw: pd.DataFrame, df_index_raw: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        df = df_stock_raw.copy()
        if FEATURE_CLOSE not in df.columns: return pd.DataFrame()
        df[FEATURE_CLOSE] = df[FEATURE_CLOSE].astype(float)
        df[FEATURE_RETURNS] = df[FEATURE_CLOSE].pct_change().astype(float)
        self.feature_names = [FEATURE_RETURNS]
        return df
    def requires_index_data(self) -> bool:
        return False # This strategy does not need index data
    

# app/feature_engineering/strategies/advanced_features.py
class ReturnsVariationStrategy(FeatureEngineeringStrategy):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.variation_window = self.config.get('variation_window', 10)
        self.min_periods_var = self.config.get('min_periods_variation', max(1, self.variation_window // 2))
        # Define feature_names produced by this strategy HERE
        self.feature_names = [FEATURE_RETURNS, FEATURE_VARIATION]

    def get_required_raw_columns(self) -> List[str]: return [FEATURE_CLOSE]

    def _calculate_base_features(self, df_stock_raw: pd.DataFrame, df_index_raw: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        df = df_stock_raw.copy()
        if FEATURE_CLOSE not in df.columns: return pd.DataFrame()
        df[FEATURE_CLOSE] = df[FEATURE_CLOSE].astype(float)
        df[FEATURE_RETURNS] = df[FEATURE_CLOSE].pct_change().astype(float)
        df[FEATURE_VARIATION] = df[FEATURE_RETURNS].rolling(
            window=self.variation_window, min_periods=self.min_periods_var
        ).std().astype(float)
        # self.feature_names is already set in __init__
        return df
    
    # requires_index_data method (returns False for this strategy)
    def requires_index_data(self) -> bool:
        return False
    


class ReturnsVarCorrStrategy(FeatureEngineeringStrategy):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.variation_window = self.config.get('variation_window', 10)
        self.min_periods_var = self.config.get('min_periods_variation', max(1, self.variation_window // 2))
        self.correlation_window = self.config.get('correlation_window', 10)
        self.min_periods_corr = self.config.get('min_periods_correlation', max(1, self.correlation_window // 2))
    def get_required_raw_columns(self) -> List[str]: return [FEATURE_CLOSE]
    def _calculate_base_features(self, df_stock_raw: pd.DataFrame, df_index_raw: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        df_s = df_stock_raw.copy()
        if FEATURE_CLOSE not in df_s.columns: return pd.DataFrame()
        df_s[FEATURE_CLOSE] = df_s[FEATURE_CLOSE].astype(float)
        df_s[FEATURE_RETURNS] = df_s[FEATURE_CLOSE].pct_change().astype(float)
        df_s[FEATURE_VARIATION] = df_s[FEATURE_RETURNS].rolling(
            window=self.variation_window, min_periods=self.min_periods_var
        ).std().astype(float)
        
        df_s[FEATURE_CORRELATION] = np.nan # Default
        if df_index_raw is not None and FEATURE_CLOSE in df_index_raw.columns: # Check if index data is provided and usable
            df_i = df_index_raw.copy()
            # Ensure index also has returns
            if FEATURE_RETURNS not in df_i.columns:
                df_i[FEATURE_CLOSE] = df_i[FEATURE_CLOSE].astype(float)
                df_i[FEATURE_RETURNS] = df_i[FEATURE_CLOSE].pct_change().astype(float)
            
            if FEATURE_RETURNS in df_i.columns: # Proceed only if index returns are available
                stock_ret = df_s[FEATURE_RETURNS][~df_s.index.duplicated(keep='last')].sort_index()
                index_ret = df_i[FEATURE_RETURNS][~df_i.index.duplicated(keep='last')].sort_index()
                aligned = pd.DataFrame({'stock_ret': stock_ret, 'index_ret': index_ret}).dropna()
                if not aligned.empty and len(aligned) >= self.min_periods_corr:
                    corr_val = aligned['stock_ret'].rolling(
                        window=self.correlation_window, min_periods=self.min_periods_corr
                    ).corr(aligned['index_ret'])
                    df_s[FEATURE_CORRELATION] = corr_val.reindex(df_s.index).astype(float)
                else:
                    logging.debug(f"Not enough aligned data for correlation for a stock. Window: {self.correlation_window}")
            else:
                logging.debug(f"Index data provided for {self.__class__.__name__} but missing '{FEATURE_RETURNS}' after processing.")
        elif self.requires_index_data() and df_index_raw is None: # Log if expected but not given
             logging.warning(f"Strategy {self.__class__.__name__} requires index data, but none was provided.")
        
        self.feature_names = [FEATURE_RETURNS, FEATURE_VARIATION, FEATURE_CORRELATION]
        return df_s

    def requires_index_data(self) -> bool:
        return True # This strategy explicitly needs index data
