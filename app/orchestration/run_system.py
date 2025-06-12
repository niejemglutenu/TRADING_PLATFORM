# trading_platform/app/orchestration/run_system.py
import logging
import os
import json
from typing import Dict, List, Optional, Any
import datetime
import pandas as pd
import time
# --- Imports from within the app package ---
from app.common.config import AppConfig # Corrected: Use the AppConfig class
from app.common.constants import (
    DEFAULT_PREDICT_DAYS, MIN_DAYS_FOR_PREDICTION_CONTEXT,
    DEFAULT_OHLCV_TABLE_NAME, FEATURE_RETURNS # Add other necessary constants
)
from app.data_ingestion.db_manager import DataManager, update_all_market_data
from app.feature_engineering.strategies import FeatureEngineeringStrategy # Import base
# Import specific strategies - this can be made more dynamic by loading based on feature_strategy_key
from app.feature_engineering.strategies import PastReturnsStrategy
from app.feature_engineering.strategies import ReturnsVariationStrategy, ReturnsVarCorrStrategy
from app.modeling.training_pipeline import train_model_pipeline # Corrected path
from app.modeling.prediction_pipeline import prediction_pipeline # Corrected path
from app.backtesting.performance import evaluate_predictions # Corrected import based on your performance.py location

logger = logging.getLogger("app.orchestration.run_system")

# REMOVED: global db_connection_pool_global_for_fetch_cache

# trading_platform/app/orchestration/run_system.py
import logging
import os
import json
from typing import Dict, List, Optional, Any
import datetime
import pandas as pd

# --- Imports from within the app package ---
from app.common.config import AppConfig # Use the AppConfig class
from app.common.constants import (
    DEFAULT_PREDICT_DAYS, MIN_DAYS_FOR_PREDICTION_CONTEXT,
    DEFAULT_OHLCV_TABLE_NAME, FEATURE_RETURNS # Add other necessary constants
)

logger = logging.getLogger("app.orchestration.run_system")


STATUS_UPDATE_PREFIX = "GUI_STATUS_UPDATE::"

def _emit_status_for_gui(status_dict: Dict[str, Any], status_file_path: str):
    """Helper to write status JSON to file for GUI to parse."""
    try:
        # Also print to stdout for easier debugging if GUI is not attached or for other consumers
        print(f"{STATUS_UPDATE_PREFIX}{json.dumps(status_dict)}", flush=True)
        # Write to file
        with open(status_file_path, 'w') as f_status:
            json.dump(status_dict, f_status, indent=2)
    except Exception as e:
        logger.warning(f"Could not emit/write status for GUI: {e}")



def run_system(
    project_root_path: str,
    mode: str,
    model_scope: str,
    tickers_to_predict: List[str],
    feature_strategy_key: str,
    feature_config_dict: Dict[str, Any], # Specific config for the chosen strategy
    training_pool_start_date: str,
    prediction_horizon: int,
    force_retrain_models: bool,
    backtest_start_date: Optional[str] = None,
    backtest_end_date: Optional[str] = None,
    force_retrain_each_step: Optional[bool] = False,
) -> Optional[Dict[str, Any]]:

    logger.info(f"--- System Run Initializing: Mode='{mode}', ModelScope='{model_scope}', Strategy='{feature_strategy_key}' ---")
    
    # --- Get Configs (unchanged from your version, looks good) ---
    db_settings = AppConfig.get('database_settings', {})
    api_settings = AppConfig.get('api_settings', {})
    data_settings_cfg = AppConfig.get('data_settings', {})
    storage_params = AppConfig.get('storage', {})

    paths = {
        "models": os.path.join(project_root_path, storage_params.get('model_artifact_path', 'data/models/')),
        "predictions": os.path.join(project_root_path, storage_params.get('local_predictions_path', 'data/predictions/')),
        "plots": os.path.join(project_root_path, storage_params.get('local_plots_path', 'data/plots/')),
        "metrics": os.path.join(project_root_path, storage_params.get('local_metrics_path', 'data/metrics/')),
        "status": os.path.join(project_root_path, storage_params.get('local_status_path', 'data/status/'))
    }
    try:
        for path_name, path_val in paths.items(): 
            os.makedirs(path_val, exist_ok=True)
            logger.debug(f"Ensured directory exists: {path_name} at {path_val}")
    except OSError as e:
        logger.error(f"Dir creation failed: {e}. Aborting."); return {"error": f"Dir creation failed: {e}"}
    
    data_manager_instance: Optional[DataManager] = None
    ohlcv_table_name = data_settings_cfg.get('ohlcv_table_name', DEFAULT_OHLCV_TABLE_NAME)

    if not db_settings or not db_settings.get("host"):
        logger.error("DB settings missing. Aborting."); return {"error": "Missing DB settings"}
    try:
        data_manager_instance = DataManager(db_settings, api_settings, ohlcv_table_name)
        if not (hasattr(data_manager_instance, 'conn_pool') and data_manager_instance.conn_pool):
            raise ConnectionError("DataManager pool not valid post-init.")
        logger.info("DataManager initialized.")
    except Exception as e:
        logger.error(f"DataManager init failed: {e}. Aborting.", exc_info=True)
        return {"error": f"DataManager init error: {e}"}

    # --- Data Update (unchanged from your version, looks good) ---
    # ... (symbols_to_update calculation and update_all_market_data call) ...
    primary_index_ticker = data_settings_cfg.get('primary_index_ticker')
    stocks_all_cfg = AppConfig.get('data_settings.STOCKS_ALL', [])
    universe_for_training = stocks_all_cfg
    default_ingest_start = data_settings_cfg.get('default_ingestion_start_date', '2018-01-01')
    timeframe = data_settings_cfg.get('default_timeframe', '1Day')
    symbols_to_update = list(set((universe_for_training or []) + (tickers_to_predict or []) + ([primary_index_ticker] if primary_index_ticker else [])))
    symbols_to_update = [s for s in symbols_to_update if s]
    if data_manager_instance and symbols_to_update:
        update_start = training_pool_start_date
        update_end = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        update_mode_cfg = AppConfig.get('data_settings.data_update_mode', "update")
        skip_api_cfg = AppConfig.get('data_settings.skip_api_for_update_globally', False)
        if mode == "backtest" and backtest_end_date:
            update_end = pd.to_datetime(backtest_end_date, utc=True).strftime("%Y-%m-%d")
            if not AppConfig.get('backtest_settings.use_api_for_data_fetch_in_backtest', True):
                skip_api_cfg = True
                update_mode_cfg = AppConfig.get('backtest_settings.data_mode_if_no_api', "ensure_table_only")
        try:
            if not update_all_market_data(symbols_to_update, db_settings, api_settings, ohlcv_table_name, timeframe,update_mode_cfg, update_start, update_end, default_ingest_start, skip_api_cfg) and update_mode_cfg not in ["ensure_table_only", "skip_api"]:
                logger.warning("Data update reported issues.")
            else: logger.info("Data update/check complete.")
        except Exception as e_data:
            logger.error(f"Data update failed: {e_data}", exc_info=True)
            if data_manager_instance: data_manager_instance.close_all_connections(); return {"error": f"Data update error: {e_data}"}

    # --- Feature Engineering Strategy (unchanged from your version, looks good) ---
    fe_instance: Optional[FeatureEngineeringStrategy] = None
    if 'lstm_window_size' not in feature_config_dict:
        feature_config_dict['lstm_window_size'] = AppConfig.get('model_settings.default_lstm_window_size', 10)
    strategy_classes = {"PastReturnsStrategy": PastReturnsStrategy, "ReturnsVariationStrategy": ReturnsVariationStrategy, "ReturnsVarCorrStrategy": ReturnsVarCorrStrategy}
    StrategyClass = strategy_classes.get(feature_strategy_key)
    if not StrategyClass:
        logger.error(f"Unknown strategy key: '{feature_strategy_key}'. Aborting.");
        if data_manager_instance: data_manager_instance.close_all_connections(); return {"error": f"Unknown strategy key: {feature_strategy_key}"}
    try:
        if StrategyClass is None:
            raise ValueError(f"StrategyClass for key '{feature_strategy_key}' is None.")
        fe_instance = StrategyClass(config=feature_config_dict)
    
        logger.info(f"Using Strategy: {fe_instance.__class__.__name__}, Config: {feature_config_dict}")
    
    
    except Exception as e_fe: # Catch error during FE instantiation         
        logger.error(f"Failed to instantiate strategy '{feature_strategy_key}': {e_fe}", exc_info=True)
        if data_manager_instance: data_manager_instance.close_all_connections(); return {"error": f"Strategy init error: {e_fe}"}
    
    if fe_instance is None: # Should have been caught by StrategyClass check
        logger.error("Critical: Feature engineering instance is None after instantiation attempt.")
        if data_manager_instance: data_manager_instance.close_all_connections()
        return {"error": "Feature engineering strategy instance None"}   
    index_ticker_for_pipelines = primary_index_ticker if fe_instance.requires_index_data() else None

    if fe_instance.requires_index_data() and not index_ticker_for_pipelines:
        logger.warning(f"Strategy {fe_instance.__class__.__name__} requires index data, but 'primary_index_ticker' is not configured or primary_index_ticker is None. Correlation-like features might be all NaN.")

    # --- Model Scope & Training Tickers (unchanged, looks good) ---
    train_tickers_list: List[str]; model_id_base: str
    if model_scope == "all_stocks_model":
        train_tickers_list = universe_for_training
        if not train_tickers_list: 
            logger.error("Training universe empty. Aborting.")

            if data_manager_instance: data_manager_instance.close_all_connections(); return {"error": "Empty training universe"}
        model_id_base = "all_stocks"
    elif model_scope == "single_stock_model":
        if not tickers_to_predict: 
            logger.error("Tickers to predict empty. Aborting.")
            if data_manager_instance: data_manager_instance.close_all_connections(); return {"error": "Empty tickers_to_predict"}
        train_tickers_list = [tickers_to_predict[0]]
        model_id_base = f"single_{train_tickers_list[0].replace('.', '_').replace('^', '')}"
    else: 
        logger.error(f"Invalid model_scope: {model_scope}. Aborting.")
        if data_manager_instance: data_manager_instance.close_all_connections(); return {"error": f"Invalid model_scope"}

    min_hist_context_for_pred = AppConfig.get('model_settings.min_days_prediction_context', MIN_DAYS_FOR_PREDICTION_CONTEXT)
    
    run_artifacts: Dict[str, Any] = {
        "run_start_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "mode": mode, "model_scope": model_scope, "strategy": feature_strategy_key,
        "tickers_predicted_for": tickers_to_predict,
        "training_pool_start_date": training_pool_start_date,
        "prediction_horizon": prediction_horizon,
    }
    if mode == "backtest":
        run_artifacts.update({"backtest_start_date": backtest_start_date, "backtest_end_date": backtest_end_date})

    status_file_path = os.path.join(paths["status"], "current_backtest_status.json")

    # --- Main Logic (Backtest or Live) ---
    try:
        if mode == "backtest":
            if not (backtest_start_date and backtest_end_date):
                raise ValueError("Backtest start and end dates are required for backtest mode.")
            
            bt_start_dt = pd.to_datetime(backtest_start_date, utc=True)
            bt_end_dt = pd.to_datetime(backtest_end_date, utc=True)

            # Generate actual business dates for iteration
            prediction_dates_for_backtest = pd.bdate_range(
                start=bt_start_dt, end=bt_end_dt, 
                tz=bt_start_dt.tz if hasattr(bt_start_dt, 'tz') else 'UTC' # Ensure tz
            )
            total_b_days_in_period = len(prediction_dates_for_backtest)

            if total_b_days_in_period <= 0:
                msg = f"No business days in specified backtest period: {bt_start_dt.date()} to {bt_end_dt.date()}."
                logger.error(msg)
                if data_manager_instance: data_manager_instance.close_all_connections()
                return {"error": msg, **run_artifacts} # Include run_artifacts
            logger.info(f"Backtest from {prediction_dates_for_backtest[0].date()} to {prediction_dates_for_backtest[-1].date()} ({total_b_days_in_period} iterations), Horizon: {prediction_horizon} days")

            overall_backtest_start_time_monotonic = time.monotonic()
            time_per_iteration_ema_sec: Optional[float] = None
            ema_alpha = 0.2 # Smoothing factor for EMA

            current_run_id_for_gui = f"bt_{model_id_base}_{fe_instance.__class__.__name__}_{prediction_dates_for_backtest[0].strftime('%Y%m%d')}_{prediction_dates_for_backtest[-1].strftime('%Y%m%d')}"
            
            # Initial status emit
            initial_status = {
                "run_id_for_gui": current_run_id_for_gui, "current_iteration": 0,
                "total_iterations_approx": total_b_days_in_period, "progress_percent": 0,
                "status_message": "Backtest initializing...", "estimated_time_remaining_str": "Calculating...",
                "overall_start_time_unix": overall_backtest_start_time_monotonic, # For GUI to calc elapsed
                "time_per_iteration_sec": None, "is_final_run_status": False
            }
            _emit_status_for_gui(initial_status, status_file_path)

            all_bt_preds_dfs: List[pd.DataFrame] = []
            
            for actual_iterations_completed, curr_pred_target_dt in enumerate(prediction_dates_for_backtest, 1):
                iter_start_time_monotonic = time.monotonic()
                train_ends_dt = curr_pred_target_dt - pd.offsets.BDay(1)
                train_ends_str = train_ends_dt.strftime('%Y-%m-%d')
                
                
                status_data_iter = initial_status.copy() # Base for this iteration
                status_data_iter["current_iteration"] = actual_iterations_completed

                status_data_iter["status_message"] = f"Iter {actual_iterations_completed}/{total_b_days_in_period}: Training (model for {train_ends_str})"
                if total_b_days_in_period > 0:
                    status_data_iter["progress_percent"] = min(100, int((actual_iterations_completed / total_b_days_in_period) * 100))


                if time_per_iteration_ema_sec and actual_iterations_completed > 1: # Can estimate after first iter done
                    remaining_iter = total_b_days_in_period - actual_iterations_completed
                    if remaining_iter >= 0:
                        eta_sec = remaining_iter * time_per_iteration_ema_sec
                        status_data_iter["estimated_time_remaining_str"] = str(datetime.timedelta(seconds=int(eta_sec)))
                    else: status_data_iter["estimated_time_remaining_str"] = "Finishing..." # Should not happen if loop logic is right
                elif actual_iterations_completed == 1: # For the first iteration, before EMA is known
                     status_data_iter["estimated_time_remaining_str"] = "Calculating (1st iter)..."
                _emit_status_for_gui(status_data_iter, status_file_path)
                
                logger.info(f"BT Iter {actual_iterations_completed}: Train to {train_ends_str}, Pred for {curr_pred_target_dt.date()}")
                model_file_id = f"bt_{model_id_base}_{fe_instance.__class__.__name__}_{train_ends_str.replace('-','')}"
                retrain_now = force_retrain_models if actual_iterations_completed == 1 else (force_retrain_each_step or False)

                m_path, x_path, y_path = train_model_pipeline(
                    train_tickers_list, index_ticker_for_pipelines, training_pool_start_date,
                    train_ends_str, model_file_id, fe_instance, data_manager_instance,
                    ohlcv_table_name, retrain_now, paths["models"]
                )
                if not all([m_path, x_path, y_path]):
                    err_msg = f"BT Model prep failed iter {actual_iterations_completed}."
                    logger.error(err_msg);
                    _emit_status_for_gui({**status_data_iter, "status_message": f"ERROR: {err_msg}", "is_final_run_status": True}, status_file_path)
                    if data_manager_instance: data_manager_instance.close_all_connections(); return {"error": err_msg, **run_artifacts}

                status_data_iter["status_message"] = f"Iter {actual_iterations_completed}/{total_b_days_in_period}: Predicting ({len(tickers_to_predict)} tickers)"
                _emit_status_for_gui(status_data_iter, status_file_path)

                for ticker_lp in tickers_to_predict:
                    pred_hist_start = (train_ends_dt - pd.Timedelta(days=min_hist_context_for_pred)).strftime('%Y-%m-%d')
                    
                    if m_path and x_path and y_path is not None:
                        preds_df_iter, _ = prediction_pipeline(
                            ticker_lp, index_ticker_for_pipelines, pred_hist_start, train_ends_str,
                            prediction_horizon, m_path, x_path, y_path, fe_instance,
                            data_manager_instance, ohlcv_table_name
                        )
                    if preds_df_iter is not None and not preds_df_iter.empty: all_bt_preds_dfs.append(preds_df_iter)
                
                iter_end_time_monotonic = time.monotonic()
                current_iter_duration = iter_end_time_monotonic - iter_start_time_monotonic
                if time_per_iteration_ema_sec is None: time_per_iteration_ema_sec = current_iter_duration
                else: time_per_iteration_ema_sec = (ema_alpha * current_iter_duration) + ((1 - ema_alpha) * time_per_iteration_ema_sec)
                
                logger.info(f"BT Iter {actual_iterations_completed} took {current_iter_duration:.2f}s. EMA iter time: {time_per_iteration_ema_sec:.2f}s")
                status_data_iter["time_per_iteration_sec"] = time_per_iteration_ema_sec
                status_data_iter["status_message"] = f"Iter {actual_iterations_completed}/{total_b_days_in_period}: Completed."
                remaining_iter_after = total_b_days_in_period - actual_iterations_completed
                if remaining_iter_after > 0 and time_per_iteration_ema_sec:
                    eta_sec_after = remaining_iter_after * time_per_iteration_ema_sec
                    status_data_iter["estimated_time_remaining_str"] = str(datetime.timedelta(seconds=int(eta_sec_after)))
                else:
                    status_data_iter["estimated_time_remaining_str"] = "Completed"
                    if remaining_iter_after <= 0: # This was the last iteration or overshot
                        status_data_iter["is_final_run_status"] = True
                        status_data_iter["progress_percent"] = 100
                _emit_status_for_gui(status_data_iter, status_file_path)
            
            # --- After the loop ---
            final_status_msg = "Backtest: Processing final results."
            if not all_bt_preds_dfs: final_status_msg = "Backtest: Loop done, no predictions generated."
            _emit_status_for_gui({
                "run_id_for_gui": current_run_id_for_gui, "current_iteration": actual_iterations_completed,
                "total_iterations_approx": total_b_days_in_period, "progress_percent": 100, 
                "status_message": final_status_msg, "estimated_time_remaining_str": "Completed",
                "overall_start_time_unix": overall_backtest_start_time_monotonic,
                "time_per_iteration_sec": time_per_iteration_ema_sec, "is_final_run_status": True
            }, status_file_path)

            if all_bt_preds_dfs:
                
                file_suffix_safe = current_run_id_for_gui # Use the consistent run_id for filenames
                final_preds_df = pd.concat(all_bt_preds_dfs, ignore_index=True)
                # ... (datetime normalization for PredictionDate, ForecastOriginDate) ...
                if 'PredictionDate' in final_preds_df.columns: final_preds_df['PredictionDate'] = pd.to_datetime(final_preds_df['PredictionDate'], errors='coerce', utc=True).dt.normalize()
                if 'ForecastOriginDate' in final_preds_df.columns: final_preds_df['ForecastOriginDate'] = pd.to_datetime(final_preds_df['ForecastOriginDate'], errors='coerce', utc=True).dt.normalize()

                preds_csv_path = os.path.join(paths["predictions"], f"predictions_{file_suffix_safe}.csv")
                relative_preds_path = os.path.relpath(preds_csv_path, os.path.join(project_root_path, "data"))
                run_artifacts["predictions_csv_path"] = relative_preds_path # e.g., "predictions/predictions_XYZ.csv"

                # For plots_output_directory
                plots_output_dir_for_run = os.path.join(paths["plots"], file_suffix_safe)
                os.makedirs(plots_output_dir_for_run, exist_ok=True)
                relative_plots_dir = os.path.relpath(plots_output_dir_for_run, os.path.join(project_root_path, "data"))
                run_artifacts["plots_output_directory"] = relative_plots_dir # e.g., "plots/run_XYZ"

                # For plot_paths_dict
                relative_plot_paths_dict = {}
                plot_p_dict = None  # Initialize to avoid unbound error
                # plot_p_dict will be set after evaluate_predictions below
                run_artifacts["plot_paths_dict"] = relative_plot_paths_dict # e.g., {"AAPL": "plots/run_XYZ/plot_AAPL.png"}
                final_preds_df.to_csv(preds_csv_path, index=False); run_artifacts["predictions_csv_path"] = preds_csv_path
                logger.info(f"Backtest predictions saved: {preds_csv_path}")

                plots_output_dir_for_run = os.path.join(paths["plots"], file_suffix_safe)
                os.makedirs(plots_output_dir_for_run, exist_ok=True)
                
                evaluation_target_col_name = fe_instance.get_target_name()
                eval_results = evaluate_predictions(
                    final_preds_df, fe_instance, data_manager_instance, ohlcv_table_name,
                    evaluation_target_col_name, plots_output_dir_for_run, f"_{file_suffix_safe}"
                )
                if eval_results: # Tuple: (merged_df, per_ticker_per_horizon, overall_per_horizon, plot_paths_dict)
                    merged_eval_df, met_tick, met_ovr, plot_p_dict = eval_results
                    if merged_eval_df is not None and not merged_eval_df.empty:
                        merged_path = os.path.join(paths["predictions"], f"merged_eval_data_{file_suffix_safe}.csv")
                        merged_eval_df.to_csv(merged_path, index=False); run_artifacts["evaluation_merged_dataframe_path"] = merged_path
                    run_artifacts.update({"plot_paths_dict": plot_p_dict, "plots_output_directory": plots_output_dir_for_run})
                    if met_ovr:
                        path = os.path.join(paths["metrics"], f"metrics_overall_horizon_{file_suffix_safe}.json")
                        with open(path, 'w') as f: json.dump(met_ovr, f, indent=4, default=str); run_artifacts["overall_horizon_metrics_path"] = path
                    if met_tick:
                        path = os.path.join(paths["metrics"], f"metrics_per_ticker_horizon_{file_suffix_safe}.json")
                        with open(path, 'w') as f: json.dump(met_tick, f, indent=4, default=str); run_artifacts["per_ticker_horizon_metrics_path"] = path
            else: logger.warning("No BT predictions generated to evaluate.")

        elif mode == "live_predict":
            live_train_ends_dt = pd.Timestamp.now(tz=datetime.timezone.utc).normalize() - pd.offsets.BDay(1)
            live_train_ends_str = live_train_ends_dt.strftime('%Y-%m-%d')
            model_id_live = f"live_{model_id_base}_{fe_instance.__class__.__name__}_{live_train_ends_str.replace('-','')}"
            logger.info(f"Starting Live Predict: Train up to {live_train_ends_str}, Horizon: {prediction_horizon} days. Model ID: {model_id_live}")
            
            m_path, x_path, y_path = train_model_pipeline(
                train_tickers_list, index_ticker_for_pipelines, training_pool_start_date,
                live_train_ends_str, model_id_live, fe_instance, data_manager_instance,
                ohlcv_table_name, force_retrain_models, paths["models"]
            )
            if not all([m_path, x_path, y_path]):
                raise RuntimeError(f"Live model prep failed for {model_id_live}.")

            all_live_preds_dfs: List[pd.DataFrame] = []
            for ticker_lp in tickers_to_predict:
                live_hist_start = (live_train_ends_dt - pd.Timedelta(days=min_hist_context_for_pred)).strftime('%Y-%m-%d')
                
                if m_path and x_path and y_path is not None:
                    preds_df_live, _ = prediction_pipeline(
                        ticker_lp, index_ticker_for_pipelines, live_hist_start, live_train_ends_str,
                        prediction_horizon, m_path, x_path, y_path, fe_instance,
                        data_manager_instance, ohlcv_table_name
                    )
                if preds_df_live is not None and not preds_df_live.empty:
                    all_live_preds_dfs.append(preds_df_live)
            
            if all_live_preds_dfs:
                final_live_df = pd.concat(all_live_preds_dfs, ignore_index=True)
                if 'PredictionDate' in final_live_df.columns:
                    final_live_df['PredictionDate'] = pd.to_datetime(final_live_df['PredictionDate'], errors='coerce', utc=True).dt.normalize()
                
                live_preds_path = os.path.join(paths["predictions"], f"predictions_{model_id_live}.csv")
                final_live_df.to_csv(live_preds_path, index=False)
                logger.info(f"Live predictions saved: {live_preds_path}")
                run_artifacts["live_predictions_csv_path"] = live_preds_path
                logger.info("Live prediction generated. Placeholder for trading engine.")
            else: logger.warning("No live predictions generated.")

        else:
            raise ValueError(f"Invalid mode: '{mode}'.")

    except Exception as e_run_main:
        logger.critical(f"System Run FAILED: {e_run_main}", exc_info=True)
        run_artifacts["error"] = str(e_run_main)
        # Attempt to emit a final error status for GUI
        final_error_status_data = {
            "run_id_for_gui": run_artifacts.get("run_id_for_gui", f"error_run_{datetime.datetime.now(datetime.timezone.utc).isoformat()}"),
            "current_iteration": run_artifacts.get("current_iteration", 0), # Try to get last known iter
            "total_iterations_approx": run_artifacts.get("total_iterations_approx", 0),
            "progress_percent": 100, # Indicate process has ended (even if in error)
            "status_message": f"ERROR: {str(e_run_main)[:250]}", # Truncate long errors
            "estimated_time_remaining_str": "Error",
            "overall_start_time_unix": run_artifacts.get("overall_start_time_unix"),
            "time_per_iteration_sec": run_artifacts.get("time_per_iteration_sec"),
            "is_final_run_status": True
        }
        _emit_status_for_gui(final_error_status_data, status_file_path)
    finally:
        if data_manager_instance:
            data_manager_instance.close_all_connections()
            logger.info("DataManager connections closed.")
            
    run_artifacts["run_end_time"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    logger.info(f"--- System Run Finished: Mode='{mode}', Scope='{model_scope}', Strategy='{feature_strategy_key}' ---")
    if "error" in run_artifacts: logger.error(f"Run finished with error: {run_artifacts['error']}")
    return run_artifacts