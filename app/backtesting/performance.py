import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
import os
from typing import Tuple, Dict, Optional, Any, List

from app.feature_engineering.strategies import FeatureEngineeringStrategy
from app.data_ingestion.db_manager import DataManager
# Constants can be imported if needed, e.g., for default values or specific column names
# from app.common.constants import FEATURE_CLOSE 

logger = logging.getLogger("app.backtesting.performance")



# from app.common.config import AppConfig # Only if AppConfig is directly used here for some reason

logger = logging.getLogger("app.backtesting.performance")

# --- Helper Metric Functions ---
def calculate_directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    pred_direction = np.sign(y_pred)
    true_direction = np.sign(y_true)
    relevant_mask = (true_direction != 0)
    if not np.any(relevant_mask): return np.nan
    matches = (pred_direction[relevant_mask] == true_direction[relevant_mask])
    return np.mean(matches) if len(matches) > 0 else np.nan

def calculate_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true_np, y_pred_np = np.array(y_true), np.array(y_pred)
    mask = y_true_np != 0
    if not np.any(mask): return np.nan
    return float(np.mean(np.abs((y_true_np[mask] - y_pred_np[mask]) / y_true_np[mask])) * 100)

def get_mda(y_true: pd.Series, y_pred: pd.Series) -> float:
    """ Mean Directional Accuracy """
    return np.mean( (np.sign(y_true) == np.sign(y_pred)).astype(int) ) * 100.0

def evaluate_predictions(
    predictions_df: pd.DataFrame, # MUST contain 'Ticker', 'PredictionDate', 'PredictedReturn', 'ForecastHorizon', 'ForecastOriginDate'
    feature_engineer: FeatureEngineeringStrategy,
    data_manager: DataManager,
    ohlcv_table_name: str,
    target_column_name: str, # Name of the target column after feature_engineer.generate_features()
    plots_output_dir: str, # Base directory for plots for this specific run
    output_suffix_for_plots: str = "" # e.g., "_run_xyz"
) -> Tuple[
        Optional[pd.DataFrame],                 # merged_df with predictions and actuals
        Optional[Dict[str, Dict[int, Dict[str, Any]]]], # per_ticker_per_horizon_metrics
        Optional[Dict[int, Dict[str, Any]]],    # overall_per_horizon_metrics
        Optional[Dict[str, str]]                # plot_paths_dict (ticker: general_plot_path)
    ]:
    """
    Evaluates predictions against actual target values, including horizon-specific metrics.
    """
    plot_paths_dict: Dict[str, str] = {}
    merged_df_final: Optional[pd.DataFrame] = None
    per_ticker_per_horizon_metrics_final: Optional[Dict[str, Dict[int, Dict[str, Any]]]] = None
    overall_per_horizon_metrics_final: Optional[Dict[int, Dict[str, Any]]] = None

    try:
        os.makedirs(plots_output_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create plots output dir '{plots_output_dir}': {e}. Plots won't be saved.")
        plots_output_dir = "" # Disable plot saving

    if predictions_df.empty or not all(col in predictions_df.columns for col in ['PredictionDate', 'Ticker', 'PredictedReturn', 'ForecastHorizon']):
        logger.warning("Evaluation: predictions_df empty or missing required columns (PredictionDate, Ticker, PredictedReturn, ForecastHorizon).")
        return None, None, None, plot_paths_dict

    logger.info(f"\n--- Evaluating Predictions (Target: '{target_column_name}', Suffix: {output_suffix_for_plots}) ---")
    try:
        predictions_df['PredictionDate'] = pd.to_datetime(predictions_df['PredictionDate'], utc=True)
        # ForecastOriginDate should also be datetime if it exists and is used
        if 'ForecastOriginDate' in predictions_df.columns:
            predictions_df['ForecastOriginDate'] = pd.to_datetime(predictions_df['ForecastOriginDate'], errors='coerce', utc=True)
    except Exception as e:
        logger.error(f"Error converting dates in predictions_df: {e}", exc_info=True)
        return None, None, None, plot_paths_dict

    unique_tickers = list(pd.unique(predictions_df['Ticker']))
    if not unique_tickers:
        logger.warning("Evaluation: No unique tickers in predictions_df."); return None, None, None, plot_paths_dict

    min_pred_date_utc = predictions_df['PredictionDate'].min()
    max_pred_date_utc = predictions_df['PredictionDate'].max()

    max_fe_lookback = max(30, feature_engineer.config.get('lstm_window_size', 10) + 20) # Simplified
    fetch_actuals_start_dt = min_pred_date_utc - pd.Timedelta(days=max_fe_lookback + 15) # Extra buffer
    fetch_actuals_end_dt = max_pred_date_utc + pd.offsets.BDay(10) # Fetch well beyond last prediction for target calculation

    logger.info(f"Fetching actuals raw data for eval: {fetch_actuals_start_dt.date()} to {fetch_actuals_end_dt.date()}")
    actuals_raw_map = data_manager.get_data_from_db(
        unique_tickers, fetch_actuals_start_dt.strftime('%Y-%m-%d'), 
        fetch_actuals_end_dt.strftime('%Y-%m-%d'), ohlcv_table_name
    )
    if not actuals_raw_map:
        logger.error("Could not fetch raw actual data."); return None, None, None, plot_paths_dict

    all_actual_targets_list = []
    for ticker, raw_df_actuals in actuals_raw_map.items():
        if raw_df_actuals.empty: continue
        required_raw_cols = feature_engineer.get_required_raw_columns()
        if not all(col in raw_df_actuals.columns for col in required_raw_cols): continue
        
        df_idx_for_target_calc = None # Assume target calc doesn't need separate index data for now
        processed_actuals = feature_engineer.generate_features(raw_df_actuals.copy(), df_idx_for_target_calc)
        
        if not processed_actuals.empty and target_column_name in processed_actuals.columns:
            actual_vals = processed_actuals[[target_column_name]].reset_index()
            idx_col = processed_actuals.index.name or 'timestamp' # Default if index is unnamed after reset
            if idx_col not in actual_vals.columns and 'index' in actual_vals.columns: idx_col = 'index'
            if idx_col not in actual_vals.columns:
                logger.error(f"Could not identify date column for actuals of {ticker}. Cols: {actual_vals.columns}"); continue
            
            actual_vals.rename(columns={idx_col: 'PredictionDate', target_column_name: 'ActualValue'}, inplace=True)
            actual_vals['Ticker'] = ticker
            all_actual_targets_list.append(actual_vals[['Ticker', 'PredictionDate', 'ActualValue']])
        else:
            logger.warning(f"No '{target_column_name}' in processed actuals for {ticker}. Cols: {processed_actuals.columns if not processed_actuals.empty else 'Empty'}")

    if not all_actual_targets_list:
        logger.error("No actual target values processed."); return None, None, None, plot_paths_dict

    all_actuals_df = pd.concat(all_actual_targets_list, ignore_index=True)
    all_actuals_df['PredictionDate'] = pd.to_datetime(all_actuals_df['PredictionDate'], errors='coerce', utc=True)
    all_actuals_df.dropna(subset=['PredictionDate'], inplace=True)

    predictions_df['PredictionDateNormalized'] = predictions_df['PredictionDate'].dt.normalize()
    all_actuals_df['PredictionDateNormalized'] = all_actuals_df['PredictionDate'].dt.normalize()
    
    merged_df = pd.merge(
        predictions_df, all_actuals_df[['Ticker', 'PredictionDateNormalized', 'ActualValue']],
        on=['Ticker', 'PredictionDateNormalized'], how='left'
    )
    merged_df.dropna(subset=['ActualValue', 'PredictedReturn'], inplace=True) # Critical for metric calculation

    if merged_df.empty:
        logger.warning("Evaluation: Merged DataFrame is empty after dropna. No matching pred/actual pairs."); return None, None, None, plot_paths_dict
    logger.info(f"Merged {len(merged_df)} predictions with actuals for metric calculation.")
    merged_df_final = merged_df.copy() # Keep a copy of the full merged data

    # --- Calculate Metrics Per Ticker, Per Horizon ---
    per_ticker_per_horizon_metrics: Dict[str, Dict[int, Dict[str, Any]]] = {}
    
    for ticker_name in merged_df['Ticker'].unique():
        per_ticker_per_horizon_metrics[ticker_name] = {}
        ticker_group_df = merged_df[merged_df['Ticker'] == ticker_name]

        for horizon_val in sorted(ticker_group_df['ForecastHorizon'].unique()):
            horizon_df_for_ticker = ticker_group_df[ticker_group_df['ForecastHorizon'] == horizon_val]
            
            y_true = horizon_df_for_ticker['ActualValue'].astype(float)
            y_pred = horizon_df_for_ticker['PredictedReturn'].astype(float)

            if len(y_true) < 1: continue # Skip if no data for this specific ticker-horizon

            metrics_for_this_set = {
                'MAE': mean_absolute_error(y_true, y_pred),
                'MSE': mean_squared_error(y_true, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'R2': r2_score(y_true, y_pred),
                'MedAE': median_absolute_error(y_true, y_pred),
                'MAPE': calculate_mape(y_true, y_pred),
                'DirectionalAccuracy': calculate_directional_accuracy(y_true, y_pred),
                'MDA_SimpleSign': get_mda(y_true, y_pred), # Simpler sign match
                'Samples': len(y_true),
                'MeanActual': y_true.mean(),
                'MeanPredicted': y_pred.mean(),
                'StdActual': y_true.std(),
                'StdPredicted': y_pred.std(),
                'Correlation': np.nan
            }
            if len(y_true) > 1 and y_true.nunique() > 1 and y_pred.nunique() > 1:
                try: metrics_for_this_set['Correlation'] = np.corrcoef(y_true, y_pred)[0, 1]
                except Exception: pass
            
            per_ticker_per_horizon_metrics[ticker_name][int(horizon_val)] = metrics_for_this_set
    per_ticker_per_horizon_metrics_final = per_ticker_per_horizon_metrics

    # --- Calculate Overall Metrics Per Horizon (Averaging over tickers) ---
    overall_per_horizon_metrics: Dict[int, Dict[str, Any]] = {}
    all_unique_horizons = sorted(list(set(h for t_data in per_ticker_per_horizon_metrics.values() for h in t_data.keys())))

    for horizon_val in all_unique_horizons:
        metrics_for_this_horizon_across_tickers = []
        for ticker_name in per_ticker_per_horizon_metrics:
            if horizon_val in per_ticker_per_horizon_metrics[ticker_name]:
                metrics_for_this_horizon_across_tickers.append(per_ticker_per_horizon_metrics[ticker_name][horizon_val])
        
        if metrics_for_this_horizon_across_tickers:
            temp_df = pd.DataFrame(metrics_for_this_horizon_across_tickers)
            averaged_metrics = {}
            for metric_key in ['MAE', 'MSE', 'RMSE', 'R2', 'MedAE', 'MAPE', 'Correlation', 'DirectionalAccuracy', 'MDA_SimpleSign']:
                if metric_key in temp_df.columns:
                    valid_values = temp_df[metric_key].dropna()
                    averaged_metrics[f'{metric_key}_avg'] = valid_values.mean() if not valid_values.empty else np.nan
            averaged_metrics['TotalSamplesAcrossTickers'] = int(temp_df['Samples'].sum())
            averaged_metrics['NumTickersWithData'] = len(temp_df)
            overall_per_horizon_metrics[int(horizon_val)] = averaged_metrics
    overall_per_horizon_metrics_final = overall_per_horizon_metrics
    
    logger.info("\n--- Per-Ticker, Per-Horizon Metrics (Sample) ---")
    for ticker, h_data in list(per_ticker_per_horizon_metrics.items())[:2]: # Log for first 2 tickers
        for h, m in h_data.items():
            logger.info(f"Ticker: {ticker}, H={h}: { {k: f'{v:.4f}' if isinstance(v, float) else v for k, v in m.items()} }")
    if overall_per_horizon_metrics:
        logger.info("\n--- Overall Average Metrics Per Horizon ---")
        for h, m in overall_per_horizon_metrics.items():
            logger.info(f"Horizon H={h}: { {k: f'{v:.4f}' if isinstance(v, float) else v for k, v in m.items()} }")


    # --- Plotting (One general plot per ticker showing all predictions vs actuals) ---
    # For Dash, you'd typically generate plots dynamically from merged_df_final.
    # These saved plots are for static reports or quick checks.
    for ticker_name_plot in merged_df['Ticker'].unique(): # Use merged_df directly for plotting data
        ticker_plot_df = merged_df[merged_df['Ticker'] == ticker_name_plot].sort_values(by='PredictionDateNormalized')
        if ticker_plot_df.empty: continue

        if plots_output_dir:
            plt.figure(figsize=(15, 7))
            plt.plot(ticker_plot_df['PredictionDateNormalized'].values, ticker_plot_df['ActualValue'].values, 
                     label=f'Actual ({target_column_name})', marker='.', linestyle='-', alpha=0.8)
            plt.plot(ticker_plot_df['PredictionDateNormalized'].values, ticker_plot_df['PredictedReturn'].values, 
                     label='Predicted Return', marker='x', linestyle='--', alpha=0.8)
            
            plt.title(f'Actual vs. Predicted Returns: {ticker_name_plot}{output_suffix_for_plots}')
            plt.xlabel('Date'); plt.ylabel('Return Value')
            plt.legend(); plt.grid(True, alpha=0.5)
            plt.xticks(rotation=45); plt.tight_layout()
            
            plot_filename = f"evaluation_plot_{ticker_name_plot.replace('.', '_').replace('^','')}{output_suffix_for_plots}.png"
            plot_abs_path = os.path.join(plots_output_dir, plot_filename)
            try:
                plt.savefig(plot_abs_path); logging.info(f"Saved plot: {plot_abs_path}")
                plot_paths_dict[ticker_name_plot] = plot_abs_path
            except Exception as e_plot: logging.error(f"Could not save plot '{plot_abs_path}': {e_plot}")
            finally: plt.close()

    return merged_df_final, per_ticker_per_horizon_metrics_final, overall_per_horizon_metrics_final, plot_paths_dict