# src/trading_system/common/constants.py

# --- Feature Name Constants ---
# These define the canonical names for features used throughout the system.
FEATURE_CLOSE = 'close'
FEATURE_OPEN = 'open'
FEATURE_HIGH = 'high'
FEATURE_LOW = 'low'
FEATURE_VOLUME = 'volume'
FEATURE_RETURNS = 'returns'
FEATURE_TARGET = 'target'       # Default name for the prediction target column
FEATURE_VARIATION = 'variation'
FEATURE_CORRELATION = 'correlation'
# Example: FEATURE_RSI = 'rsi'
# Example: FEATURE_MACD = 'macd'

# --- Default Operational Parameters ---
# These are sensible defaults if not specified in any configuration.
# For specific runs, these should ideally be set in a config file or via CLI.

DEFAULT_PREDICT_DAYS = 2
MIN_DAYS_FOR_PREDICTION_CONTEXT = 60 # Min history for initializing prediction sequences
DEFAULT_LSTM_WINDOW_SIZE = 10        # A very common default if a strategy doesn't specify

# --- Default Configuration Keys/Values (Used by main_cli.py if config is missing sections) ---
MODE_DEFAULT = "backtest"
MODEL_SCOPE_DEFAULT = "single_stock_model"
FEATURE_STRATEGY_KEY_DEFAULT = "ReturnsVariationStrategy" # Default strategy to use
LOG_LEVEL_DEFAULT = "INFO"

# --- Default File/Path Names (Less common to override via CLI, more via main config) ---
DEFAULT_MODEL_SUBDIR = "models"
DEFAULT_CACHE_SUBDIR = "cache"
DEFAULT_LOGS_SUBDIR = "logs"
DEFAULT_REPORTS_SUBDIR = "reports"
DEFAULT_TEMPLATES_SUBDIR = "templates" # For report templates

# --- Default Evaluation Target ---
# This defines what column in the "actuals" data the predictions are compared against.
# It should align with what your models are trained to predict (self.target_name in strategies).
DEFAULT_EVALUATION_TARGET_COLUMN = FEATURE_RETURNS # If predicting next day's raw return
# DEFAULT_EVALUATION_TARGET_COLUMN = FEATURE_TARGET # If target is pre-shifted (less common for direct eval)

# --- Default Index Ticker (can be overridden in config) ---
DEFAULT_INDEX_TICKER_SYMBOL = "NDAQ" # e.g., NASDAQ Composite
# DEFAULT_INDEX_TICKER_SYMBOL = "SPY"   # e.g., S&P 500 ETF

# --- Default Table Names (can be overridden in config) ---
DEFAULT_OHLCV_TABLE_NAME = "NAS100"

# --- Default Column Names from Raw Data Sources (if you have a mapping layer) ---
# RAW_SOURCE_TIMESTAMP_COL = 't'
# RAW_SOURCE_CLOSE_COL = 'c'
# (Your fetch_and_cache_data_incrementally already expects 'timestamp', 'close', etc.
#  If fetching from diverse APIs, you might have a mapping step before caching).