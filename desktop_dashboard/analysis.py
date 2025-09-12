import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

logger = logging.getLogger("desktop_dashboard.analysis")

def calculate_directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    if y_true.empty or y_pred.empty or len(y_true) != len(y_pred):
        return np.nan
    
    try:
        pred_direction = np.sign(y_pred)
        true_direction = np.sign(y_true)
        relevant_mask = (true_direction != 0)
        if not np.any(relevant_mask):
            return np.nan
        matches = (pred_direction[relevant_mask] == true_direction[relevant_mask])
        return float(np.mean(matches)) if len(matches) > 0 else np.nan
    except Exception as e:
        logger.error(f"Error calculating directional accuracy: {e}")
        return np.nan

def calculate_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    if y_true.empty or y_pred.empty or len(y_true) != len(y_pred):
        return np.nan
    
    try:
        mask = (y_true != 0)
        if not np.any(mask):
            return np.nan
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))
    except Exception as e:
        logger.error(f"Error calculating MAPE: {e}")
        return np.nan

def calculate_risk_metrics(returns: pd.Series) -> Dict[str, float]:
    if returns.empty:
        return {}
    
    try:
        metrics = {}
        returns = returns.dropna()
        if len(returns) < 2:
            return {}
            
        # Basic statistics
        metrics['MeanReturn'] = float(returns.mean())
        metrics['StdDev'] = float(returns.std())
        metrics['Skewness'] = float(stats.skew(returns))
        metrics['Kurtosis'] = float(stats.kurtosis(returns))
        
        # Risk metrics
        metrics['SharpeRatio'] = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
        metrics['SortinoRatio'] = float(returns.mean() / returns[returns < 0].std() * np.sqrt(252)) if len(returns[returns < 0]) > 0 else 0
        
        # Drawdown metrics
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        metrics['MaxDrawdown'] = float(drawdowns.min())
        metrics['AvgDrawdown'] = float(drawdowns[drawdowns < 0].mean()) if len(drawdowns[drawdowns < 0]) > 0 else 0
        
        return metrics
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {e}")
        return {}

def analyze_predictions(predictions_df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    if predictions_df.empty:
        return {}, {}
    
    try:
        per_ticker_metrics = {}
        overall_metrics = {}
        
        # Group by ticker
        for ticker, ticker_df in predictions_df.groupby('Ticker'):
            ticker_metrics = {}
            
            # Calculate basic metrics
            y_true = ticker_df['ActualValue'].astype(float)
            y_pred = ticker_df['PredictedReturn'].astype(float)
            
            if len(y_true) > 0 and len(y_pred) > 0:
                ticker_metrics.update({
                    'MAE': mean_absolute_error(y_true, y_pred),
                    'MSE': mean_squared_error(y_true, y_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                    'R2': r2_score(y_true, y_pred),
                    'MedAE': median_absolute_error(y_true, y_pred),
                    'MAPE': calculate_mape(y_true, y_pred),
                    'DirectionalAccuracy': calculate_directional_accuracy(y_true, y_pred),
                    'Samples': len(y_true),
                    'MeanActual': y_true.mean(),
                    'MeanPredicted': y_pred.mean(),
                    'StdActual': y_true.std(),
                    'StdPredicted': y_pred.std(),
                    'Correlation': y_true.corr(y_pred) if len(y_true) > 1 else np.nan
                })
                
                # Add risk metrics
                ticker_metrics.update(calculate_risk_metrics(y_true))
                
                per_ticker_metrics[ticker] = ticker_metrics
        
        # Calculate overall metrics
        if per_ticker_metrics:
            overall_metrics = {
                'TotalSamples': sum(m['Samples'] for m in per_ticker_metrics.values()),
                'NumTickers': len(per_ticker_metrics),
                'AvgMAE': np.mean([m['MAE'] for m in per_ticker_metrics.values()]),
                'AvgRMSE': np.mean([m['RMSE'] for m in per_ticker_metrics.values()]),
                'AvgR2': np.mean([m['R2'] for m in per_ticker_metrics.values()]),
                'AvgDirectionalAccuracy': np.mean([m['DirectionalAccuracy'] for m in per_ticker_metrics.values()])
            }
        
        return per_ticker_metrics, overall_metrics
        
    except Exception as e:
        logger.error(f"Error analyzing predictions: {e}")
        return {}, {}

def prepare_plot_data(predictions_df: pd.DataFrame, ticker: str) -> Optional[Dict[str, Any]]:
    if predictions_df.empty:
        return None
        
    try:
        ticker_df = predictions_df[predictions_df['Ticker'] == ticker]
        if ticker_df.empty:
            return None
            
        # Get metrics for this ticker
        per_ticker_metrics, _ = analyze_predictions(ticker_df)
        metrics = per_ticker_metrics.get(ticker, {})
        
        return {
            'dates': ticker_df['PredictionDate'].tolist(),
            'actual_returns': ticker_df['ActualValue'].tolist(),
            'predicted_returns': ticker_df['PredictedReturn'].tolist(),
            'metrics': metrics
        }
    except Exception as e:
        logger.error(f"Error preparing plot data for {ticker}: {e}")
        return None 