# trading_platform/desktop_dashboard/data_loader.py
import pandas as pd
import json
import os
import logging
from typing import Dict, Optional, Any, List

logger = logging.getLogger("desktop_dashboard.data_loader")

# Determine artifact paths relative to this data_loader.py file's location
# Assumes data_loader.py is in desktop_dashboard/, and project_root is one level up.
try:
    GUI_APP_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(GUI_APP_DIR, "..")) # trading_platform/
    
    ARTIFACT_BASE_DIR = os.path.join(PROJECT_ROOT, "data")
    PREDICTIONS_DIR = os.path.join(ARTIFACT_BASE_DIR, "predictions")
    METRICS_DIR = os.path.join(ARTIFACT_BASE_DIR, "metrics")
    PLOTS_DIR = os.path.join(ARTIFACT_BASE_DIR, "plots") # Base directory for all run-specific plot folders
    STATUS_DIR = os.path.join(ARTIFACT_BASE_DIR, "status")
    LOGS_DIR = os.path.join(PROJECT_ROOT, "logs") # Assuming logs are at project_root/logs

    # Create directories if they don't exist (especially for first-time GUI use)
    for dir_path in [ARTIFACT_BASE_DIR, PREDICTIONS_DIR, METRICS_DIR, PLOTS_DIR, STATUS_DIR, LOGS_DIR]:
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
            except OSError as e:
                logger.error(f"Could not create directory {dir_path}: {e}")
    
    logger.info(f"DataLoader initialized. Project Root: {PROJECT_ROOT}, Artifact Base: {ARTIFACT_BASE_DIR}")

except Exception as e:
    logger.critical(f"Critical error setting up essential paths in data_loader: {e}", exc_info=True)
    # Set to dummy paths to allow import but signal failure
    PROJECT_ROOT, ARTIFACT_BASE_DIR, PREDICTIONS_DIR, METRICS_DIR, PLOTS_DIR, STATUS_DIR, LOGS_DIR = \
        ".", ".", ".", ".", ".", ".", "."
    raise # Re-raise to make it clear that path setup failed


def get_list_of_backtest_runs() -> List[str]:
    """Scans the predictions directory for completed backtest run artifact identifiers."""
    runs = []
    if not os.path.exists(PREDICTIONS_DIR) or not os.path.isdir(PREDICTIONS_DIR):
        logger.warning(f"Predictions directory not found or not a directory: {PREDICTIONS_DIR}")
        return runs
    try:
        for f_name in os.listdir(PREDICTIONS_DIR):
            # Assuming predictions files are named "predictions_<run_id>.csv"
            # And run_id starts with "bt_"
            if f_name.startswith("predictions_bt_") and f_name.endswith(".csv"):
                run_id = f_name[len("predictions_"):-len(".csv")]
                runs.append(run_id)
    except Exception as e:
        logger.error(f"Error scanning predictions directory {PREDICTIONS_DIR}: {e}", exc_info=True)
    
    if not runs:
        logger.info(f"No completed backtest runs found in {PREDICTIONS_DIR}")
    return sorted(list(set(runs)), reverse=True)


def load_run_data(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Loads all artifacts for a given backtest run_id.
    The run_id is the suffix used in filenames (e.g., "bt_single_AAPL_..._dates").
    """
    data: Dict[str, Any] = {'run_id': run_id, 'plot_files': {}} # ticker: absolute_host_path
    logger.info(f"Loading artifact data for backtest run_id: {run_id}")

    if not run_id or not isinstance(run_id, str):
        logger.warning(f"load_run_data called with invalid run_id: {run_id}")
        return None

    found_any_data = False

    try:
        # 1. Predictions CSV
        preds_file = os.path.join(PREDICTIONS_DIR, f"predictions_{run_id}.csv")
        if os.path.exists(preds_file):
            try:
                df = pd.read_csv(preds_file)
                for col in ['PredictionDate', 'ForecastOriginDate']: # Convert known date columns
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
                data['predictions_df'] = df
                found_any_data = True
                logger.debug(f"Loaded predictions for {run_id} from {preds_file}: {len(df)} rows.")
            except Exception as e_csv:
                 logger.error(f"Error reading or processing predictions CSV {preds_file}: {e_csv}")
        else:
            logger.warning(f"Predictions file not found: {preds_file}")

        # 2. Overall Metrics JSON
        overall_metrics_file = os.path.join(METRICS_DIR, f"metrics_overall_horizon_{run_id}.json")
        if os.path.exists(overall_metrics_file):
            try:
                with open(overall_metrics_file, 'r', encoding='utf-8') as f:
                    data['overall_horizon_metrics'] = json.load(f)
                found_any_data = True
                logger.debug(f"Loaded overall metrics for {run_id} from {overall_metrics_file}.")
            except Exception as e_json_ovr:
                 logger.error(f"Error reading or parsing overall metrics JSON {overall_metrics_file}: {e_json_ovr}")
        else:
            logger.warning(f"Overall metrics file not found: {overall_metrics_file}")
        
        # 3. Per-Ticker Metrics JSON
        ticker_metrics_file = os.path.join(METRICS_DIR, f"metrics_per_ticker_horizon_{run_id}.json")
        if os.path.exists(ticker_metrics_file):
            try:
                with open(ticker_metrics_file, 'r', encoding='utf-8') as f:
                    data['per_ticker_horizon_metrics'] = json.load(f)
                found_any_data = True
                logger.debug(f"Loaded per-ticker metrics for {run_id} from {ticker_metrics_file}.")
            except Exception as e_json_tck:
                logger.error(f"Error reading or parsing per-ticker metrics JSON {ticker_metrics_file}: {e_json_tck}")
        else:
            logger.warning(f"Per-ticker metrics file not found: {ticker_metrics_file}")
        
        # 4. Plot files
        # `evaluate_predictions` saves plots to: PLOTS_DIR / <run_id> / evaluation_plot_TICKER_<run_id>.png
        # `run_id` here is the `file_suffix_safe` from `run_system`, which is `current_run_id_for_gui`
        plots_subdir_for_this_run = os.path.join(PLOTS_DIR, run_id) # e.g., data/plots/bt_single_AAPL_..._dates/
        
        if os.path.isdir(plots_subdir_for_this_run):
            plot_count = 0
            for plot_filename_on_host in os.listdir(plots_subdir_for_this_run):
                # Filename format: evaluation_plot_TICKERSANITIZED_RUNID.png
                # where RUNID is the same as our input `run_id`
                prefix = "eval_plot_"
                # The suffix used by evaluate_predictions is output_suffix_for_plots which is f"_{run_id}"
                expected_filename_suffix_component = f"_{run_id}.png"

                if plot_filename_on_host.startswith(prefix) and \
                   plot_filename_on_host.endswith(expected_filename_suffix_component):
                    
                    # Extract ticker: it's between "eval_plot_" and "_{run_id}.png"
                    ticker_in_filename = plot_filename_on_host[len(prefix) : -len(expected_filename_suffix_component)]
                    # The ticker in filename might have been sanitized (e.g. BRK.B -> BRK_B)
                    # For now, we assume this sanitized ticker is the key we want.
                    # If you need to map it back to original (e.g. "BRK.B"), you'd need a list of original tickers from predictions_df.
                    
                    absolute_plot_path = os.path.join(plots_subdir_for_this_run, plot_filename_on_host)
                    data['plot_files'][ticker_in_filename] = absolute_plot_path
                    plot_count += 1
                    logger.debug(f"DataLoader: Found plot for ticker '{ticker_in_filename}': {absolute_plot_path}")
            logger.info(f"Found {plot_count} plot files for run '{run_id}' in '{plots_subdir_for_this_run}'.")
            if plot_count > 0 : found_any_data = True
        else:
            logger.warning(f"Plots subdirectory not found for run '{run_id}' at host path: '{plots_subdir_for_this_run}'")

        return data if found_any_data else None # Return None if absolutely no artifacts were found for this run_id

    except Exception as e:
        logger.error(f"General error in load_run_data for run_id '{run_id}': {e}", exc_info=True)
        return None


def load_live_status() -> Optional[Dict[str, Any]]:
    """Loads the current backtest status from `current_backtest_status.json`."""
    status_file = os.path.join(STATUS_DIR, "current_backtest_status.json")
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from status file: {status_file}. File might be corrupt or being written.")
            return None
        except Exception as e:
            logger.error(f"Error reading status file {status_file}: {e}", exc_info=True)
            return None
    # logger.debug(f"Status file not found: {status_file}") # Can be noisy if polled frequently
    return None # File not found or not readable