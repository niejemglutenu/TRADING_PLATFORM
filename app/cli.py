# trading_platform/app/cli.py
import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.common.constants import setup_all_directories, get_all_portfolio_strategies
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
except Exception as e_path:
    print(f"ERROR (cli.py): Could not set up sys.path: {e_path}")
    sys.exit(1)
try:
    from app.common.config import AppConfig
    from app.orchestration.run_system import run_system
except ModuleNotFoundError as e:
    print(f"ERROR (cli.py): Module Not Found. Details: {e}")
    sys.exit(1)
def str_to_bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def parse_arguments():
    parser = argparse.ArgumentParser(description='Trading System CLI', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    core_group = parser.add_argument_group('Core Settings')
    core_group.add_argument('--base-config', type=str, help='Base configuration file.')
    core_group.add_argument('--profile-config', type=str, help='Profile configuration file.')
    core_group.add_argument('--mode', type=str, default='backtest', choices=['backtest', 'live_predict'], help='Operation mode.')
    core_group.add_argument('--run-id', type=str, required=True, help='A specific run ID to use for artifacts.')
    core_group.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Logging level.')
    data_group = parser.add_argument_group('Data and Timeframe')
    data_group.add_argument('--tickers', type=str, nargs='+', required=True, help='List of tickers to process.')
    data_group.add_argument('--training-start-date', type=str, required=True, help='Start date for training data (YYYY-MM-DD).')
    data_group.add_argument('--backtest-start-date', type=str, required=True, help='Start date for backtest (YYYY-MM-DD).')
    data_group.add_argument('--backtest-end-date', type=str, required=True, help='End date for backtest (YYYY-MM-DD).')
    model_group = parser.add_argument_group('ML Model and Prediction')
    model_group.add_argument('--feature-strategy', type=str, required=False, help='Feature engineering strategy (required for predictive strategies).')
    model_group.add_argument('--model-scope', type=str, default='all_stocks_model', choices=['single_stock_model', 'all_stocks_model'], help='Model training scope.')
    model_group.add_argument('--prediction-horizon', type=int, default=2, help='Number of days into the future to predict.')
    model_group.add_argument('--model', type=str, default='LSTM_Shuffle', choices=['LSTM_Shuffle', 'LSTM_NoShuffle', 'Transformer'], help='ML model architecture.')
    model_group.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    model_group.add_argument('--batch-size', type=int, default=32, help='Batch size for training.')
    model_group.add_argument('--load-predictions-from-run', type=str, default=None, help='Load a prediction file from a previous run, skipping all ML steps.')
    train_group = parser.add_argument_group('Training Strategy')
    train_group.add_argument('--load-model-id', type=str, default=None, help='Load a pre-trained model ID, skipping initial training.')
    train_group.add_argument('--save-model-as', type=str, default=None, help='Custom tag for any newly trained models.')
    train_group.add_argument('--force-retrain', action='store_true', help='Force retraining of the initial model.')
    train_group.add_argument('--force-retrain-steps', action='store_true', help='Force retraining at every single step of the backtest.')
    train_group.add_argument('--retrain-frequency', type=int, default=0, help='Retrain every N days. 0 means train only once.')
    portfolio_group = parser.add_argument_group('Portfolio and Risk Management')
    portfolio_group.add_argument('--portfolio-strategy', type=str, default='MarkowitzHistoric', choices=get_all_portfolio_strategies(), help='The portfolio construction strategy to use.')
    portfolio_group.add_argument('--rebalance-days', type=int, default=5, help='The number of days between portfolio rebalancing.')
    portfolio_group.add_argument('--lookback-period', type=int, default=252, help='Historical lookback for strategy calculations.')
    portfolio_group.add_argument('--top-k', type=int, default=10, help='Number of top stocks to consider.')
    portfolio_group.add_argument('--max-position-size', type=float, default=0.25, help='Maximum weight for any single position.')
    portfolio_group.add_argument('--min-position-size', type=float, default=0.01, help='Minimum weight for any single position.')
    portfolio_group.add_argument('--entry-threshold', type=float, default=0.01, help='Minimum predicted return to consider a stock.')
    portfolio_group.add_argument('--allow-shorting', type=str_to_bool, nargs='?', const=True, default=False, help='Allow short positions.')
    portfolio_group.add_argument('--fully-invested', type=str_to_bool, nargs='?', const=True, default=True, help='Ensure portfolio is fully invested (no cash).')
    risk_group = parser.add_argument_group('Advanced Risk and Optimization')
    risk_group.add_argument('--use-shrinkage', type=str_to_bool, nargs='?', const=True, default=False, help='Use shrinkage estimation for covariance.')
    risk_group.add_argument('--enable-stop-loss-take-profit', type=str_to_bool, nargs='?', const=True, default=False, help='Enable stop-loss and take-profit orders.')
    risk_group.add_argument('--use-trailing-stop', type=str_to_bool, nargs='?', const=True, default=False, help='Enable trailing stop orders.')
    risk_group.add_argument('--stop-loss-pct', type=float, default=0.05, help='Stop loss percentage from entry price.')
    risk_group.add_argument('--take-profit-pct', type=float, default=0.15, help='Take profit percentage from entry price.')
    risk_group.add_argument('--trailing-stop-pct', type=float, default=0.03, help='Trailing stop percentage from high-water mark.')
    risk_group.add_argument('--market-ticker', type=str, default='SPY', help='Market ticker for benchmark-relative calculations.')
    return parser.parse_args()
def main():
    try:
        args = parse_arguments()
        setup_all_directories()
        root_logger = logging.getLogger()
        if root_logger.hasHandlers():
            root_logger.handlers.clear()
        logger = logging.getLogger("app.cli")
        logger.info("Initializing AppConfig...")
        AppConfig.initialize(
            project_root=PROJECT_ROOT,
            base_config_filename=args.base_config,
            profile_config_filename=args.profile_config
        )
        config = AppConfig.get_instance()
        fully_invested = args.fully_invested 
        logger.info(f"Starting system run with ID: {args.run_id}")
        results = run_system(
            run_id=args.run_id,
            mode=args.mode,
            tickers_to_predict=args.tickers,
            training_pool_start_date=args.training_start_date,
            backtest_start_date=args.backtest_start_date,
            backtest_end_date=args.backtest_end_date,
            feature_strategy_key=args.feature_strategy,
            feature_config_dict=config.get('feature_engineering', {}).get('strategies_config', {}).get(args.feature_strategy, {}) if args.feature_strategy else {},
            model_scope=args.model_scope,
            prediction_horizon=args.prediction_horizon,
            model=args.model,
            load_model_id=args.load_model_id,
            save_model_as=args.save_model_as,
            force_retrain=args.force_retrain,
            force_retrain_steps=args.force_retrain_steps,
            retrain_frequency=args.retrain_frequency, 
            load_predictions_from_run=args.load_predictions_from_run,
            portfolio_strategy=args.portfolio_strategy,
            top_k=args.top_k,
            allow_shorting=args.allow_shorting,
            fully_invested=fully_invested,  
            max_position_size=args.max_position_size,  
            min_position_size=args.min_position_size,  
            lookback_period=args.lookback_period,
            rebalance_days=args.rebalance_days,
            epochs=args.epochs,
            batch_size=args.batch_size,
            entry_threshold=args.entry_threshold,
            market_ticker=args.market_ticker,
            enable_stop_loss_take_profit=args.enable_stop_loss_take_profit,
            stop_loss_pct=args.stop_loss_pct,
            take_profit_pct=args.take_profit_pct,
            trailing_stop_pct=args.trailing_stop_pct,
            use_trailing_stop=args.use_trailing_stop,
        )
        if results and not results.get("error"):
            logger.info("System run completed successfully.")
        else:
            logger.error(f"System run failed. Results: {results}")
            sys.exit(1)
    except Exception as e:
        logging.getLogger("app.cli").error(f"A critical error occurred in main: {e}", exc_info=True)
        sys.exit(1)
if __name__ == '__main__':
    main()