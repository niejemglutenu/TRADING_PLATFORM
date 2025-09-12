# trading_platform/app/orchestration/run_system.py
import logging
import os
import json
from typing import Dict, List, Optional, Any, Type
from datetime import timedelta, datetime, timezone
import pandas as pd
import time
from app.common.config import AppConfig
from app.common.constants import (
    get_models_dir, get_raw_predictions_filepath, get_merged_eval_filepath,
    get_metrics_filepath, get_predictive_strategies, get_historic_strategies, 
    is_predictive_strategy, is_historic_strategy
)
from app.data_ingestion.db_manager import DataManager
from app.feature_engineering.strategies import FeatureEngineeringStrategy
from app.modeling.training_pipeline import train_model_pipeline
from app.modeling.prediction_pipeline import prediction_pipeline
from app.backtesting.performance import evaluate_and_backtest
from app.backtesting.backtest import (
    MarkowitzHistoric, 
    MarkowitzPredicted,
    EnhancedMarkowitzPredicted, 
    MarkowitzHistoricEfficientReturn, 
    MinSemiVarianceHistoric, 
    MeanCVaRHistoric,
    TopKPredicted,
    MinSemiVariancePredicted,
    MinCVaRPredicted,
    PredictiveMomentumFilter
)
import hashlib
import re
import sys



logger = logging.getLogger("app.orchestration.run_system")
STATUS_UPDATE_PREFIX = "GUI_STATUS_UPDATE::"

def setup_logging(level=logging.INFO):
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    if not root_logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s: %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

def _emit_status_for_gui(status_data: Dict[str, Any]):
    if "progress_percent" not in status_data:
        total = status_data.get("total_iterations_approx", 1)
        current = status_data.get("current_iteration", 0)
        status_data["progress_percent"] = min(100, int((current / total) * 100)) if total > 0 else 0
    print(f"{STATUS_UPDATE_PREFIX}{json.dumps(status_data, default=str)}")



def _generate_model_identifier(
    feature_strategy_name: str,
    model_name: str, 
    num_stocks: int,
    prediction_horizon: int,
    train_start_date_str: str,
    train_end_date_str: str,
    custom_tag: Optional[str] = None
) -> str:
 
    
    safe_fe_name = re.sub(r'[^a-zA-Z0-9_-]+', '', feature_strategy_name)
    safe_start_date = train_start_date_str.replace('-', '')
    safe_end_date = train_end_date_str.replace('-', '')
    
    parts = [
        safe_fe_name,
        model_name,
        f"{num_stocks}stocks",
        f"h{prediction_horizon}", # Use 'h' for horizon
        f"{safe_start_date}-to-{safe_end_date}"
    ]

    if custom_tag:
        safe_custom_tag = re.sub(r'[\s\W]+', '_', custom_tag).strip('_')
        if safe_custom_tag:
            parts.append(safe_custom_tag)

    return "__".join(parts)


class ETAEstimator:
    """A helper class to manage and calculate the estimated time remaining."""
    def __init__(self):
        self.time_for_train_step: Optional[float] = None
        self.predict_step_times: List[float] = []

    def record_duration(self, duration: float, is_train_step: bool):
        if is_train_step:
            self.time_for_train_step = duration
        else:
            self.predict_step_times.append(duration)
            if len(self.predict_step_times) > 5:
                self.predict_step_times.pop(0)

    def get_eta_str(self, remaining_steps: int, num_future_trains: int) -> Optional[str]:
        if remaining_steps <= 0: return None
        avg_predict_time = sum(self.predict_step_times) / len(self.predict_step_times) if self.predict_step_times else None
        if not self.time_for_train_step and not avg_predict_time: return None
        est_train_time = self.time_for_train_step or (avg_predict_time * 10.0)
        est_predict_time = avg_predict_time or (self.time_for_train_step / 10.0)
        num_future_predicts = remaining_steps - num_future_trains
        eta_seconds = (num_future_trains * est_train_time) + (num_future_predicts * est_predict_time)
        return str(timedelta(seconds=int(eta_seconds)))


def _generate_backtest_predictions(
    run_id: str,
    fe_instance: FeatureEngineeringStrategy,
    data_manager: DataManager,
    config: Dict,
    cli_args: Dict
) -> List[pd.DataFrame]:
  
    backtest_start_date = cli_args['backtest_start_date']
    backtest_end_date = cli_args['backtest_end_date']
    tickers_to_predict = cli_args['tickers_to_predict']
    model_scope = cli_args['model_scope']
    prediction_horizon = cli_args['prediction_horizon']
    training_pool_start_date = cli_args['training_pool_start_date']
    load_model_id = cli_args.get('load_model_id')
    save_model_as_tag = cli_args.get('save_model_as')
    force_retrain_initial = cli_args.get('force_retrain', False)
    force_retrain_each_step = cli_args.get('force_retrain_steps', False)
    retrain_frequency = int(cli_args.get('retrain_frequency', 0))
    model_name = cli_args.get('model', 'LSTM_Shuffle') # THE FIX: Correct dict access
    epochs = cli_args.get('epochs', 50)
    batch_size = cli_args.get('batch_size', 32)

    data_settings_cfg = config['data_settings']
    ohlcv_table_name = data_settings_cfg['ohlcv_table_name']
    train_tickers_list = data_settings_cfg.get('STOCKS_ALL', []) if model_scope == "all_stocks_model" else tickers_to_predict
    index_ticker_for_pipelines = data_settings_cfg.get('primary_index_ticker') if fe_instance.requires_index_data() else None
    min_hist_context_for_pred = config.get('model_settings', {}).get('min_days_prediction_context', 100)

    prediction_dates = pd.bdate_range(start=backtest_start_date, end=backtest_end_date, tz='UTC')
    all_preds_dfs: List[pd.DataFrame] = []
    days_since_last_train = 0
    current_model_path = None
    

    iteration_times: List[float] = []



    if load_model_id := cli_args.get('load_model_id'):
        logger.info(f"Using single pre-trained model for all predictions: {load_model_id}")
        current_model_path = str(get_models_dir() / load_model_id)
        if not os.path.isdir(current_model_path):
            raise FileNotFoundError(f"Specified model to load not found: {current_model_path}")

    for i, current_decision_date in enumerate(prediction_dates, 1):
        iter_start_time = time.monotonic()
        train_end_date = current_decision_date - pd.offsets.BDay(1)
        
        status_data = {
            "run_id_for_gui": run_id,
            "current_iteration": i,
            "total_iterations_approx": len(prediction_dates)
        }        
        should_train_this_step = False
        if not load_model_id:
            if i == 1 or cli_args.get('force_retrain_steps', False) or (int(cli_args.get('retrain_frequency', 0)) > 0 and days_since_last_train >= int(cli_args.get('retrain_frequency', 0))):
                should_train_this_step = True
                

        

        
        if should_train_this_step:
            status_data["status_message"] = f"Iter {i}: Training..."
            _emit_status_for_gui(status_data)

            model_identifier = _generate_model_identifier(
                feature_strategy_name=fe_instance.__class__.__name__,
                model_name=model_name, 
                num_stocks=len(train_tickers_list),
                prediction_horizon=prediction_horizon,
                train_start_date_str=training_pool_start_date,
                train_end_date_str=train_end_date.strftime('%Y-%m-%d'),
                custom_tag=save_model_as_tag
            )
            
            model_base_path = str(get_models_dir() / run_id) if force_retrain_each_step else str(get_models_dir())

            trained_path = train_model_pipeline(
                tickers_for_training=train_tickers_list,
                index_ticker_symbol=index_ticker_for_pipelines,
                training_data_start_date_str=training_pool_start_date,
                training_data_end_date_str=train_end_date.strftime('%Y-%m-%d'),
                model_identifier_str=model_identifier,
                feature_engineer=fe_instance,
                data_manager=data_manager,
                ohlcv_table_name=ohlcv_table_name,
                force_retrain=force_retrain_initial or should_train_this_step,
                model_artifacts_base_path=model_base_path,
                model=model_name,
                epochs=epochs,
                batch_size=batch_size,
                prediction_horizon=prediction_horizon
            )
            
            if not trained_path:
                raise RuntimeError(f"Model training failed for iteration {i}.")
            
            current_model_path = trained_path
            
            days_since_last_train = 0
        else:
            days_since_last_train += 1

        if not current_model_path:
            raise RuntimeError("No model is available for prediction. Check training/loading logic.")

        status_data["status_message"] = f"Iter {i}: Predicting..."
        _emit_status_for_gui(status_data)

        logger.info(f"Iteration {i}: Generating predictions for {len(cli_args['tickers_to_predict'])} tickers")
        predictions_generated = 0

        for ticker in cli_args['tickers_to_predict']:
            pred_hist_start = (train_end_date - pd.Timedelta(days=min_hist_context_for_pred)).strftime('%Y-%m-%d')
            
            try:
                logger.info(f"Generating prediction for {ticker} using data from {pred_hist_start} to {train_end_date.strftime('%Y-%m-%d')}")
                
                preds_df, _ = prediction_pipeline(
                    ticker_to_predict=ticker,
                    index_ticker_symbol=index_ticker_for_pipelines,
                    historical_data_fetch_start_date_str=pred_hist_start, # USE THE CORRECTED START DATE
                    historical_data_end_date_str=train_end_date.strftime('%Y-%m-%d'),
                    n_days_to_predict=prediction_horizon,
                    model_load_path=current_model_path,
                    feature_engineer=fe_instance,
                    data_manager=data_manager,
                    ohlcv_table_name=ohlcv_table_name
                )
                
                if preds_df is not None and not preds_df.empty:
                    all_preds_dfs.append(preds_df)
                    predictions_generated += 1
                    logger.info(f"Successfully generated {len(preds_df)} predictions for {ticker}")
                else:
                    logger.warning(f"No predictions generated for {ticker} - likely missing historical data")
                    
            except Exception as e:
                logger.warning(f"Failed to generate predictions for {ticker}: {e}")
                continue
        
        logger.info(f"Iteration {i}: Generated predictions for {predictions_generated}/{len(cli_args['tickers_to_predict'])} tickers")
        
        
        # --- Final Status Update Block ---
        iter_duration = time.monotonic() - iter_start_time
        if i <= 2:
            iteration_times.append(iter_duration)
        
        status_data = {
            "run_id_for_gui": run_id,
            "current_iteration": i,
            "total_iterations_approx": len(prediction_dates),
            "status_message": f"Iter {i}: Completed"
        }
        
        if i == 1:
            status_data['time_for_first_iter'] = iteration_times[0]
        elif i == 2:
            status_data['time_for_first_iter'] = iteration_times[0]
            status_data['time_for_second_iter'] = iteration_times[1]



        status_data['days_since_last_train'] = days_since_last_train
        status_data["status_message"] = f"Iter {i}: Completed."

        _emit_status_for_gui(status_data)

    return all_preds_dfs
def _run_backtest_mode(
    run_id: str,
    data_manager: DataManager,
    fe_instance: Optional[FeatureEngineeringStrategy],
    config: Dict,
    cli_args: Dict
) -> Dict:

    PREDICTIVE_STRATEGIES = get_predictive_strategies()
    HISTORIC_STRATEGIES = get_historic_strategies()

    logger.info(f"--- Running Backtest Mode ---")
    
    run_artifacts = {}
    final_preds_df = pd.DataFrame()
    portfolio_strategy_key = cli_args['portfolio_strategy']

    load_preds_run_id = cli_args.get('load_predictions_from_run')
    if load_preds_run_id:
        logger.info(f"Loading predictions from previous run: {load_preds_run_id}")
        preds_filepath = get_raw_predictions_filepath(load_preds_run_id)
        if preds_filepath.exists():
            final_preds_df = pd.read_csv(preds_filepath)
            logger.info(f"Unique tickers: {sorted(final_preds_df['Ticker'].unique())}")
            logger.info(f"Null values: {final_preds_df.isnull().sum().to_dict()}")
            run_artifacts['loaded_predictions_from_run'] = load_preds_run_id
        else:
            raise FileNotFoundError(f"Could not find prediction file: {preds_filepath}")
    else:
        if is_predictive_strategy(portfolio_strategy_key):
            if fe_instance is None:
                raise ValueError(f"Feature engineering strategy is required for predictive strategy '{portfolio_strategy_key}' but none was provided.")
            logger.info("Portfolio strategy requires predictions. Starting prediction pipeline...")
            all_preds_dfs = _generate_backtest_predictions(run_id, fe_instance, data_manager, config, cli_args)
            if all_preds_dfs:
                final_preds_df = pd.concat(all_preds_dfs, ignore_index=True)
                preds_path = get_raw_predictions_filepath(run_id)
                final_preds_df.to_csv(preds_path, index=False)
                run_artifacts['raw_predictions_path'] = str(preds_path)
                logger.info(f"New predictions saved: {preds_path}")
            else:
                logger.warning("No predictions were generated. This may cause issues with predictive strategies.")
                final_preds_df = pd.DataFrame()
        else:
            logger.info(f"Portfolio strategy '{portfolio_strategy_key}' is historic. Skipping prediction generation.")
            

    if is_historic_strategy(portfolio_strategy_key):
        try:
            train_start = pd.to_datetime(cli_args['training_pool_start_date'])
            backtest_start = pd.to_datetime(cli_args['backtest_start_date'])
            lookback_period = len(pd.bdate_range(train_start, backtest_start, inclusive='left'))
            logger.info(f"Dynamic lookback period calculated: {lookback_period} days (from {train_start} to {backtest_start})")
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid dates for dynamic lookback: {e}. Falling back to default.")
            lookback_period = 252
    else:
        lookback_period = int(cli_args.get('lookback_period', 252))
    
    logger.info(f"Using lookback period of {lookback_period} days for strategy {portfolio_strategy_key}.")


    strategy_map = {
        'MarkowitzHistoric': MarkowitzHistoric,
        'MarkowitzHistoricEfficientReturn': MarkowitzHistoricEfficientReturn,
        'MinSemiVarianceHistoric': MinSemiVarianceHistoric,
        'MeanCVaRHistoric': MeanCVaRHistoric,
        'MarkowitzPredicted': MarkowitzPredicted,
        'EnhancedMarkowitzPredicted': EnhancedMarkowitzPredicted,
        'TopKPredicted': TopKPredicted,
        'MinSemiVariancePredicted': MinSemiVariancePredicted,
        'MinCVaRPredicted': MinCVaRPredicted,
        'PredictiveMomentumFilter': PredictiveMomentumFilter,

    } 
    PortfolioStrategyClass = strategy_map.get(portfolio_strategy_key)
    


    logger.info("Assembling a complete dictionary of all strategy parameters from CLI arguments...")
    
    strategy_params = cli_args.copy()
    
    
    strategy_params['predictions_df'] = final_preds_df
    strategy_params['backtest_start'] = cli_args['backtest_start_date']
    int_keys = ['rebalance_days', 'top_k', 'epochs', 'batch_size', 'prediction_horizon', 'lookback_period']
    float_keys = ['max_position_size', 'min_position_size', 'entry_threshold', 'stop_loss_pct', 'take_profit_pct', 'trailing_stop_pct']
    bool_keys = ['enable_stop_loss_take_profit', 'use_trailing_stop', 'allow_shorting', 'fully_invested']
    date_keys = ['backtest_start']
    
    for key in int_keys:
        if key in strategy_params and strategy_params[key] is not None:
            strategy_params[key] = int(strategy_params[key])
            
    for key in float_keys:
        if key in strategy_params and strategy_params[key] is not None:
            strategy_params[key] = float(strategy_params[key])
            
    for key in bool_keys:
        if key in strategy_params and strategy_params[key] is not None:
            strategy_params[key] = bool(strategy_params[key])

    for key in date_keys:
        if key in strategy_params and strategy_params[key] is not None:
            strategy_params[key] = str(strategy_params[key])

    logger.info(f"Passing final strategy parameters to backtrader: {list(strategy_params.keys())}")
    
 
    merged_eval_df, overall_metrics = evaluate_and_backtest(
        run_id=run_id,
        predictions_df=final_preds_df,
        data_manager=data_manager,
        ohlcv_table_name=config['data_settings']['ohlcv_table_name'],
        backtest_start_date=cli_args['backtest_start_date'],
        backtest_end_date=cli_args['backtest_end_date'],
        tickers_for_bt=cli_args['tickers_to_predict'],
        portfolio_strategy_class=PortfolioStrategyClass,
        portfolio_strategy_params=strategy_params)

    if overall_metrics:
        metrics_path = get_metrics_filepath(run_id)
        with open(metrics_path, 'w') as f: json.dump(overall_metrics, f, indent=4, default=str)
        run_artifacts["overall_metrics_path"] = str(metrics_path)
        logger.info(f"Overall metrics saved to: {metrics_path}")

    if merged_eval_df is not None and not merged_eval_df.empty:
        merged_path = get_merged_eval_filepath(run_id)
        merged_eval_df.to_csv(merged_path, index=False)
        run_artifacts["evaluation_merged_dataframe_path"] = str(merged_path)
        logger.info(f"Merged evaluation data saved to: {merged_path}")
    return run_artifacts


def run_system(**kwargs: Any) -> Optional[Dict[str, Any]]:

    setup_logging(logging.INFO)

    mode = kwargs.get('mode')
    run_id = kwargs.get('run_id')
    logger.info(f"--- System Run Initializing: Mode='{mode}', RunID='{run_id}' ---")


    run_artifacts = {"run_id": run_id, "run_start_time": datetime.now(timezone.utc).isoformat(), "parameters": {k: v for k, v in kwargs.items() if k not in ['project_root_path']}}
    data_manager_instance = None
    try:
        config = AppConfig.get_instance()
        data_manager_instance = DataManager(config['database_settings'], config.get('api_settings', {}), config['data_settings']['ohlcv_table_name'])
        
        fe_instance = None
        if kwargs.get('feature_strategy_key'):
            fe_instance = FeatureEngineeringStrategy.create(kwargs['feature_strategy_key'], kwargs.get('feature_config_dict', {}))
        
        if mode == "backtest":
            backtest_artifacts = _run_backtest_mode(run_id=run_id, data_manager=data_manager_instance, fe_instance=fe_instance, config=config, cli_args=kwargs)
            run_artifacts.update(backtest_artifacts)
        elif mode == "live_predict":
            live_artifacts = _run_live_predict_mode(**kwargs)
            run_artifacts.update(live_artifacts)
        else:
            raise ValueError(f"Invalid mode: '{mode}'.")
    except Exception as e_run_main:
        logger.critical(f"System Run FAILED: {e_run_main}", exc_info=True)
        run_artifacts["error"] = str(e_run_main)
        _emit_status_for_gui({"run_id_for_gui": run_id or "error_run", "status_message": f"ERROR: {str(e_run_main)[:250]}", "is_final_run_status": True})
    finally:
        if data_manager_instance:
            data_manager_instance.close_all_connections()
            logger.info("DataManager connections closed.")
            
    run_artifacts["run_end_time"] = datetime.now(timezone.utc).isoformat()
    logger.info(f"--- System Run Finished ---")
    return run_artifacts



def _run_live_predict_mode(**kwargs: Any) -> Dict:
    # This is a placeholder for your future live trading logic
    logger.info("Live prediction mode is not  implemented.")
    return {"status": "Live prediction not implemented."}

