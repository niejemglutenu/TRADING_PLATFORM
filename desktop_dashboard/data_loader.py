# trading_platform/desktop_dashboard/data_loader.py
import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd

# --- Setup Project Root and Imports from the App ---
# This ensures the dashboard can find the application's common code
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in os.sys.path:
        os.sys.path.insert(0, str(PROJECT_ROOT))

    from app.common.constants import (
        get_metrics_dir,
        get_merged_eval_filepath,
        get_metrics_filepath,
        get_equity_curve_filepath,
        get_plots_dir,
        get_raw_predictions_filepath
    )
except ImportError as e:
    raise ImportError(f"Could not import from 'app.common.constants'. Ensure PYTHONPATH is correct. Error: {e}")
from app.common.constants import RAW_PREDICTIONS_DIR # Import the directory constant

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("desktop_dashboard.data_loader")

# --- Helper Functions ---
def _ensure_file_readable(file_path: Path) -> bool:
    """Checks if a file exists and is readable, logging an error if not."""
    if not file_path.exists():
        logger.warning(f"File does not exist: {file_path}")
        return False
    if not os.access(file_path, os.R_OK):
        logger.warning(f"File is not readable: {file_path}")
        return False
    return True

# --- Main Data Loading Functions ---

def get_list_of_backtest_runs() -> List[str]:
    """
    Scans the metrics directory to find all completed backtest runs
    using the new, simplified naming convention.
    """
    try:
        metrics_dir = get_metrics_dir()
        if not metrics_dir.exists():
            logger.warning(f"Metrics directory does not exist: {metrics_dir}")
            return []

        # Find all metrics files (e.g., metrics_run01_... .json)
        # and extract the run_id from their names.
        run_ids = [file.stem.replace('metrics_', '') for file in metrics_dir.glob("metrics_*.json")]
        
        # Sort by most recent first (assuming run IDs have a timestamp or incrementing number)
        return sorted(run_ids, reverse=True)
    
    except Exception as e:
        logger.error(f"Error getting list of backtest runs: {e}", exc_info=True)
        return []

# In trading_platform/desktop_dashboard/data_loader.py
# In trading_platform/desktop_dashboard/data_loader.py

def load_run_data(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Loads all data artifacts for a specific backtest run.
    This version correctly constructs the final data dictionary.
    """
    if not run_id:
        return None

    try:
        # ======================================================================
        # THE FIX: Start with an empty dictionary and populate it.
        # ======================================================================
        final_run_data: Dict[str, Any] = {'run_id': run_id}

        # --- 1. Load Metrics JSON ---
        metrics_file = get_metrics_filepath(run_id)
        if _ensure_file_readable(metrics_file):
            with open(metrics_file, 'r') as f:
                # Load the nested dictionary from the JSON
                metrics_json_content = json.load(f)
                # Add the contents of the JSON to our main dictionary
                final_run_data.update(metrics_json_content)
        else:
            # If the main metrics file is missing, we can't proceed.
            logger.error(f"Cannot load run data: Main metrics file not found for run_id: {run_id}")
            return None

        # --- 2. Load Merged Predictions CSV ---
        merged_eval_file = get_merged_eval_filepath(run_id)
        if _ensure_file_readable(merged_eval_file):
            final_run_data['merged_eval_df'] = pd.read_csv(merged_eval_file)
        else:
            final_run_data['merged_eval_df'] = pd.DataFrame()

        # --- 3. Load Equity Curve CSV ---
        equity_curve_file = get_equity_curve_filepath(run_id)
        if _ensure_file_readable(equity_curve_file):
            final_run_data['equity_curve_df'] = pd.read_csv(equity_curve_file)
        else:
            final_run_data['equity_curve_df'] = pd.DataFrame()
        
        return final_run_data
        
    except Exception as e:
        logger.error(f"Failed to load data for run '{run_id}': {e}", exc_info=True)
        return None
        
