# trading_platform/app/backtesting/plotting.py

import logging
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def save_equity_curve_plot(df: pd.DataFrame, save_path: Path):
    if df.empty:
        logger.warning(f"Equity curve DataFrame is empty. Skipping plot generation for {save_path}.")
        return

    logger.info(f"Generating equity curve plot and saving to {save_path}")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    df['Date'] = pd.to_datetime(df['Date'])
    ax.plot(df['Date'], df['PortfolioValue'], label='Equity Curve', color='blue', linewidth=2)
    
    ax.set_title('Portfolio Equity Curve', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    fig.tight_layout()
    
    try:
        # Ensure the parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
    except Exception as e:
        logger.error(f"Failed to save equity curve plot to {save_path}: {e}")
    finally:
        plt.close(fig) # Important to free up memory

# In trading_platform/app/backtesting/plotting.py

def save_prediction_plots(eval_df: pd.DataFrame, run_id: str, get_plot_filepath_func):
    """
    ### REVISED ###
    Generates and saves prediction-related plots for EVERY ticker in the evaluation data.
    """
    if eval_df.empty:
        logger.warning("Evaluation DataFrame is empty. Skipping prediction plots.")
        return
        
    # Loop through each unique ticker found in the results
    for ticker in eval_df['Ticker'].unique():
        logger.info(f"--- Generating prediction plots for ticker: {ticker} ---")
        
        df = eval_df[eval_df['Ticker'] == ticker].copy()
        if df.empty:
            continue

        df['PredictionDate'] = pd.to_datetime(df['PredictionDate'])
        df.sort_values('PredictionDate', inplace=True)
        df['residual'] = df['ActualReturn'] - df['PredictedReturn']

        # --- 1. Time Series Plot ---
        # The plot name now includes the ticker symbol for uniqueness
        path1 = get_plot_filepath_func(run_id, f'timeseries_{ticker}')
        logger.info(f"Saving time series plot to {path1}")
        fig1, ax1 = plt.subplots(figsize=(12, 7))
        ax1.plot(df['PredictionDate'], df['ActualReturn'], label='Actual', color='blue', marker='o', linestyle='--')
        ax1.plot(df['PredictionDate'], df['PredictedReturn'], label='Predicted', color='red', marker='x', linestyle='-')
        ax1.set_title(f'Predicted vs. Actual Returns for {ticker}', fontsize=16)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Return', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()
        fig1.tight_layout()
        path1.parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(path1, dpi=100, bbox_inches='tight')
        plt.close(fig1)

        # --- 2. Scatter Plot ---
        path2 = get_plot_filepath_func(run_id, f'scatter_{ticker}')
        logger.info(f"Saving scatter plot to {path2}")
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.scatter(df['ActualReturn'], df['PredictedReturn'], alpha=0.7, edgecolors='k')
        min_val = min(df['ActualReturn'].min(), df['PredictedReturn'].min())
        max_val = max(df['ActualReturn'].max(), df['PredictedReturn'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit')
        ax2.set_title(f'Scatter Plot for {ticker}', fontsize=16)
        ax2.set_xlabel('Actual Return', fontsize=12)
        ax2.set_ylabel('Predicted Return', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()
        fig2.tight_layout()
        path2.parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(path2, dpi=100, bbox_inches='tight')
        plt.close(fig2)

        # --- 3. Residuals Plot ---
        path3 = get_plot_filepath_func(run_id, f'residuals_{ticker}')
        logger.info(f"Saving residuals plot to {path3}")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.scatter(df['PredictedReturn'], df['residual'], alpha=0.7, edgecolors='k', color='purple')
        ax3.axhline(0, color='r', linestyle='--', label='Zero Error')
        ax3.set_title(f'Residuals Plot for {ticker}', fontsize=16)
        ax3.set_xlabel('Predicted Return', fontsize=12)
        ax3.set_ylabel('Residual (Actual - Predicted)', fontsize=12)
        ax3.grid(True, linestyle='--', alpha=0.6)
        ax3.legend()
        fig3.tight_layout()
        path3.parent.mkdir(parents=True, exist_ok=True)
        fig3.savefig(path3, dpi=100, bbox_inches='tight')
        plt.close(fig3)