# trading_platform/app/common/constants.py
import os
from pathlib import Path
import logging

# ==============================================================================
# 1. CORE PROJECT PATHS
# ==============================================================================
# Assumes this file is in trading_platform/app/common/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "configs"

# ==============================================================================
# 2. DATA SUBDIRECTORIES (Single Source of Truth)
# ==============================================================================
DB_DIR = DATA_DIR / "db"
MODELS_DIR = DATA_DIR / "models"
METRICS_DIR = DATA_DIR / "metrics"
PLOTS_DIR = DATA_DIR / "plots"
RAW_PREDICTIONS_DIR = DATA_DIR / "raw_predictions"
MERGED_EVAL_DIR = DATA_DIR / "merged_evaluation_data"
EQUITY_CURVE_DIR = DATA_DIR / "equity_curve_data"

# ==============================================================================
# 3. DIRECTORY & PATH HELPER FUNCTIONS (Consistent Naming)
# ==============================================================================
def get_plots_dir() -> Path:
    """Gets the path for the main plots directory."""
    return PLOTS_DIR
def get_models_dir() -> Path:
    return MODELS_DIR

def get_metrics_dir() -> Path:
    return METRICS_DIR

def get_raw_predictions_filepath(run_id: str) -> Path:
    """Gets the path for the raw (unmerged) predictions CSV file."""
    return RAW_PREDICTIONS_DIR / f"predictions_{run_id}.csv"

def get_merged_eval_filepath(run_id: str) -> Path:
    """Gets the path for the merged evaluation data CSV file."""
    return MERGED_EVAL_DIR / f"merged_eval_{run_id}.csv"

def get_metrics_filepath(run_id: str) -> Path:
    """Gets the path for the main metrics JSON file for a given run."""
    return METRICS_DIR / f"metrics_{run_id}.json"

def get_plot_filepath(run_id: str, plot_name: str) -> Path:
    """Gets the path for a specific plot image for a given run."""
    return PLOTS_DIR / f"plot_{plot_name}_{run_id}.png"

def get_equity_curve_filepath(run_id: str) -> Path:
    """Returns the path to the equity curve CSV for a given run."""
    return EQUITY_CURVE_DIR / f"equity_curve_{run_id}.csv"

# ==============================================================================
# 4. DIRECTORY SETUP UTILITY
# ==============================================================================
def setup_all_directories():
    """Creates all necessary data directories if they don't exist."""
    dirs_to_create = [
        DATA_DIR, LOG_DIR, DB_DIR, MODELS_DIR, METRICS_DIR, PLOTS_DIR,
        RAW_PREDICTIONS_DIR, MERGED_EVAL_DIR, EQUITY_CURVE_DIR
    ]
    for dir_path in dirs_to_create:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logging.getLogger(__name__).error(f"Error creating directory {dir_path}: {e}")

# ==============================================================================
# 5. DEFAULT OPERATIONAL PARAMETERS
# ==============================================================================
DEFAULT_OHLCV_TABLE_NAME = "ohlcv_data"
MIN_DAYS_FOR_PREDICTION_CONTEXT = 100
DEFAULT_INITIAL_CAPITAL = 100000.0
STATUS_UPDATE_PREFIX = "GUI_STATUS_UPDATE::"

# ==============================================================================
# 6. STRATEGY REGISTRY (Single Source of Truth)
# ==============================================================================

# All available portfolio strategies
PORTFOLIO_STRATEGIES = {
    # Historic strategies (based on historical data only)
    'MarkowitzHistoric': 'MarkowitzHistoric',
    'MarkowitzHistoricEfficientReturn': 'MarkowitzHistoricEfficientReturn', 
    'MinSemiVarianceHistoric': 'MinSemiVarianceHistoric',
    'MeanCVaRHistoric': 'MeanCVaRHistoric',
    
    # Predictive strategies (require ML predictions)
    'MarkowitzPredicted': 'MarkowitzPredicted',
    'EnhancedMarkowitzPredicted': 'EnhancedMarkowitzPredicted',
    'TopKPredicted': 'TopKPredicted',
    'MinSemiVariancePredicted': 'MinSemiVariancePredicted',
    'MinCVaRPredicted': 'MinCVaRPredicted',
    'PredictiveMomentumFilter': 'PredictiveMomentumFilter'
}

# Strategy classifications for easy filtering
HISTORIC_STRATEGIES = [
    'MarkowitzHistoric',
    'MarkowitzHistoricEfficientReturn', 
    'MinSemiVarianceHistoric',
    'MeanCVaRHistoric'
]

PREDICTIVE_STRATEGIES = [
    'MarkowitzPredicted',
    'EnhancedMarkowitzPredicted',
    'TopKPredicted',
    'MinSemiVariancePredicted',
    'MinCVaRPredicted',
    'PredictiveMomentumFilter'
]

# All strategy classes (including base and intermediate classes)
ALL_STRATEGY_CLASSES = {
    # Base classes
    'PortfolioStrategy': 'PortfolioStrategy',
    'HistoricPortfolioStrategy': 'HistoricPortfolioStrategy',
    'PredictedPortfolioStrategy': 'PredictedPortfolioStrategy',
    'MarketAwareHistoricStrategy': 'MarketAwareHistoricStrategy',
    
    # Concrete historic strategies
    'MarkowitzHistoric': 'MarkowitzHistoric',
    'MarkowitzHistoricEfficientReturn': 'MarkowitzHistoricEfficientReturn', 
    'MinSemiVarianceHistoric': 'MinSemiVarianceHistoric',
    'MeanCVaRHistoric': 'MeanCVaRHistoric',
    
    # Concrete predictive strategies
    'MarkowitzPredicted': 'MarkowitzPredicted',
    'EnhancedMarkowitzPredicted': 'EnhancedMarkowitzPredicted',
    'PredictiveMomentumFilter': 'PredictiveMomentumFilter'
}

# Feature engineering strategies
FEATURE_ENGINEERING_STRATEGIES = [
    'PastReturnsStrategy',
    'ReturnsVariationStrategy', 
    'ReturnsRelativeStrengthStrategy',
    'TechnicalIndicatorsForReturn',
    'TechnicalIndicatorsForSharpe'
]

# Helper functions for strategy validation
def is_historic_strategy(strategy_name: str) -> bool:
    """Check if a strategy is historic (doesn't require predictions)."""
    return strategy_name in HISTORIC_STRATEGIES

def is_predictive_strategy(strategy_name: str) -> bool:
    """Check if a strategy is predictive (requires ML predictions)."""
    return strategy_name in PREDICTIVE_STRATEGIES

def get_all_portfolio_strategies() -> list:
    """Get all available portfolio strategy names."""
    return list(PORTFOLIO_STRATEGIES.keys())

def get_historic_strategies() -> list:
    """Get all historic strategy names."""
    return HISTORIC_STRATEGIES.copy()

def get_predictive_strategies() -> list:
    """Get all predictive strategy names."""
    return PREDICTIVE_STRATEGIES.copy()

def get_feature_engineering_strategies() -> list:
    """Get all feature engineering strategy names."""
    return FEATURE_ENGINEERING_STRATEGIES.copy()

def validate_portfolio_strategy(strategy_name: str) -> bool:
    """Validate if a strategy name is valid."""
    return strategy_name in PORTFOLIO_STRATEGIES

def validate_feature_strategy(strategy_name: str) -> bool:
    """Validate if a feature engineering strategy name is valid."""
    return strategy_name in FEATURE_ENGINEERING_STRATEGIES

def get_all_strategy_classes() -> dict:
    """Get all strategy classes including base and intermediate classes."""
    return ALL_STRATEGY_CLASSES.copy()

def get_concrete_strategies() -> list:
    """Get only the concrete (implemented) strategy names (excluding base classes)."""
    return list(PORTFOLIO_STRATEGIES.keys())

def get_base_strategy_classes() -> list:
    """Get base and intermediate strategy class names."""
    return [
        'PortfolioStrategy',
        'HistoricPortfolioStrategy', 
        'PredictedPortfolioStrategy',
        'MarketAwareHistoricStrategy'
    ]

def is_base_strategy_class(strategy_name: str) -> bool:
    """Check if a strategy name is a base/intermediate class."""
    return strategy_name in get_base_strategy_classes()

def is_concrete_strategy(strategy_name: str) -> bool:
    """Check if a strategy name is a concrete (implemented) strategy."""
    return strategy_name in PORTFOLIO_STRATEGIES