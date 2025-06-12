# scripts/initial_model_trainer.py
import logging
import os
import sys
# Add src to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path: sys.path.insert(0, SRC_PATH)
import pandas as pd
from app.modeling.training_pipeline import train_model_pipeline
from app.feature_engineering.strategies import ReturnsVarCorrStrategy # Example
from app.common.config import AppConfig
from app.data_ingestion.db_manager import DataManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    CONFIG_PATH = "configs/dev_config.yaml" # Path to your config
    app_cfg = AppConfig.get_instance()
    if not app_cfg:
        logging.error("Failed to load config for initial training.")
        sys.exit(1)

    db_pool = DataManager(db_settings= app_cfg['database_settings'], api_config= app_cfg['api_settings'])
    if not db_pool:
        sys.exit(1)

    # --- Define parameters for the initial model ---
    initial_training_tickers = app_cfg['data_parameters']['universe_symbols_for_ingestion']
    initial_index_ticker = app_cfg['data_parameters']['index_ticker']
    
    # Example: Train up to end of last month
    today = pd.Timestamp.now(tz='UTC').normalize()
    initial_training_end_dt = (today - pd.offsets.MonthBegin(1)) - pd.offsets.BDay(1) 
    initial_training_end_str = initial_training_end_dt.strftime('%Y-%m-%d')
    initial_training_start_str = app_cfg['run_parameters']['training_pool_start_date']

    strategy_key = app_cfg['feature_parameters']['strategy_class_name']
    strategy_cfg = app_cfg['feature_parameters']['strategy_configs'].get(strategy_key, {})
    strategy_cfg['lstm_window_size'] = app_cfg['model_parameters'].get('lstm_window_size', 10)

    initial_feature_engineer = None
    if strategy_key == "ReturnsVarCorrStrategy": # Example
        initial_feature_engineer = ReturnsVarCorrStrategy(config=strategy_cfg)
    # Add other strategies
    
    if not initial_feature_engineer:
        logging.error(f"Could not instantiate strategy: {strategy_key}")
        DataManager.close_all_connections(db_pool)
        sys.exit(1)

    model_id = f"initial_{strategy_key}_{initial_training_end_str.replace('-','')}"
    
    logging.info(f"Starting initial model training: ID={model_id}, EndDate={initial_training_end_str}")

    train_model_pipeline(
        tickers_for_training=initial_training_tickers,
        index_ticker_symbol=initial_index_ticker,
        training_data_start_date_str=initial_training_start_str,
        training_data_end_date_str=initial_training_end_str,
        model_identifier_str=model_id,
        feature_engineer=initial_feature_engineer,
        force_retrain=True, # Crucial for the first run
        data_manager=db_pool,
        ohlcv_table_name=app_cfg['data_parameters']['ohlcv_table_name']
    )
    logging.info("Initial model training complete.")
    DataManager.close_all_connections(db_pool)