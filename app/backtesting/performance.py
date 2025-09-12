# trading_platform/app/backtesting/performance.py
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, Any, List, Type
import backtrader as bt
import json
from datetime import datetime, timezone
from sklearn.metrics import mean_absolute_error, r2_score
from app.data_ingestion.db_manager import DataManager
from backtrader.analyzers import SharpeRatio, Returns, DrawDown, TradeAnalyzer, TimeReturn, PyFolio
from app.common.constants import get_equity_curve_filepath

logger = logging.getLogger("app.backtesting.performance")

# --- BACKTESTING CORE ---

def get_backtrader_analyzers() -> List[Tuple[Any, str]]:
    return [
        (SharpeRatio, {'_name': 'sharpe', 'riskfreerate': 0.0, 'annualize': True}),
        (Returns, {'_name': 'returns'}),
        (DrawDown, {'_name': 'drawdown'}),
        (TradeAnalyzer, {'_name': 'trades'}),
        (PyFolio, {'_name': 'pyfolio'})
    ]

def extract_backtrader_metrics(strategy_instance: bt.Strategy, run_id: str, backtest_start_date: datetime, rebalance_days: int) -> Dict[str, Any]:
    analyzers = strategy_instance.analyzers
    broker = strategy_instance.broker
    metrics = {}
    metrics['FinalPortfolioValue'] = broker.getvalue()
    metrics['InitialPortfolioValue'] = broker.startingcash
    if hasattr(analyzers, 'sharpe') and (sharpe_analysis := analyzers.sharpe.get_analysis()):
        metrics['SharpeRatio'] = sharpe_analysis.get('sharperatio')
    if hasattr(analyzers, 'returns') and (returns_analysis := analyzers.returns.get_analysis()):
        metrics['TotalReturnPct'] = returns_analysis.get('rtot', 0.0) * 100
        metrics['AnnualizedReturnPct'] = returns_analysis.get('rnorm', 0.0) * 100
    if hasattr(analyzers, 'drawdown') and (dd_analysis := analyzers.drawdown.get_analysis()):
        max_dd = dd_analysis.get('max', {})
        metrics['MaxDrawdownPct'] = max_dd.get('drawdown', 0.0)
        metrics['MaxDrawdownLen'] = max_dd.get('len', 0)
    if hasattr(analyzers, 'trades') and (trades_analysis := analyzers.trades.get_analysis()):
        total_trades = trades_analysis.get('total', {}).get('total', 0)
        metrics['TotalTrades'] = total_trades
        if total_trades > 0:
            won_trades_data = trades_analysis.get('won', {})
            lost_trades_data = trades_analysis.get('lost', {})
            won_total = won_trades_data.get('total', 0)
            lost_total = lost_trades_data.get('total', 0)
            metrics.update({
                'WonTrades': won_total,
                'LostTrades': lost_total,
                'WinRatePct': (won_total / total_trades) * 100 if total_trades > 0 else 0,
                'AvgWinTrade': won_trades_data.get('pnl', {}).get('average', 0.0),
                'AvgLossTrade': lost_trades_data.get('pnl', {}).get('average', 0.0),
            })
            won_pnl_total = won_trades_data.get('pnl', {}).get('total', 0.0)
            lost_pnl_total = lost_trades_data.get('pnl', {}).get('total', 0.0)
            if lost_pnl_total != 0:
                metrics['ProfitFactor'] = abs(won_pnl_total / lost_pnl_total)
    if hasattr(analyzers, 'trades'):
        trades_analysis = analyzers.trades.get_analysis()
        closed_trades = []
        for trade_list in trades_analysis.get('closed', {}).values():
            for trade in trade_list:
                closed_trades.append(trade)
        closed_trades_sorted = sorted(closed_trades, key=lambda t: t['pnl'], reverse=True)
        metrics['Top5Wins'] = closed_trades_sorted[:5]
        metrics['Top5Losses'] = closed_trades_sorted[-5:]
        per_stock_pnl = {}
        for trade in closed_trades:
            ticker = trade['ticker']
            per_stock_pnl[ticker] = per_stock_pnl.get(ticker, 0) + trade['pnl']
        metrics['PerStockPnL'] = per_stock_pnl
    final_holdings = {}
    total_value = broker.getvalue()
    for data in strategy_instance.datas:
        ticker = data._name
        position_size = broker.getposition(data).size
        if position_size != 0:
            position_value = broker.getposition(data).size * data.close[0]
            final_holdings[ticker] = {
                'shares': position_size,
                'value': position_value,
                'weight_pct': (position_value / total_value) * 100
            }
    cash_value = broker.getcash()
    final_holdings['Cash'] = {
        'shares': None,
        'value': cash_value,
        'weight_pct': (cash_value / total_value) * 100
    }
    metrics['FinalHoldings'] = final_holdings
    if hasattr(analyzers, 'pyfolio') and (pyfolio_analysis := analyzers.pyfolio.get_analysis()):
        try:
            returns_dict = pyfolio_analysis.get('returns', {})
            if returns_dict:
                returns_series = pd.Series(returns_dict)
            else:
                returns_series = pd.Series(dtype=float)
            if not returns_series.empty:
                initial_value = broker.startingcash
                cumulative_growth = (1 + returns_series).cumprod()
                equity_curve = initial_value * cumulative_growth
                equity_data_polished = equity_curve.to_frame(name='PortfolioValue').reset_index()
                equity_data_polished.rename(columns={'index': 'Date'}, inplace=True)
                final_calculated_value = equity_data_polished['PortfolioValue'].iloc[-1]
                actual_final_value = broker.getvalue()
                logger.info(f"Equity curve calculation verification:")
                logger.info(f"  Final calculated value: ${final_calculated_value:,.2f}")
                logger.info(f"  Actual final value: ${actual_final_value:,.2f}")
                logger.info(f"  Difference: ${abs(final_calculated_value - actual_final_value):,.2f}")
                if abs(final_calculated_value - actual_final_value) > 1.0:
                    logger.warning(f"Discrepancy found in equity curve calculation!")
                equity_filepath = get_equity_curve_filepath(run_id)
                equity_data_polished.to_csv(equity_filepath, index=False)
                logger.info(f"Saved correct equity curve data to: {equity_filepath}")
            else:
                logger.warning("PyFolio analyzer returned no daily returns data.")
        except Exception as e:
            logger.error(f"Failed to process equity curve data from PyFolio analyzer: {e}", exc_info=True)
    return metrics

def _calculate_prediction_metrics(merged_df: pd.DataFrame) -> Tuple[Dict, Dict]:
    if merged_df.empty:
        return {}, {}
    def calculate_metrics_for_group(group: pd.DataFrame) -> Dict:
        if len(group) < 2:
            return {'prediction_count': len(group)}
        actual = group['ActualReturn'].values
        predicted = group['PredictedReturn'].values
        mae = mean_absolute_error(actual, predicted)
        r2_score_val = r2_score(actual, predicted)
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        epsilon = 1e-8
        mape = np.mean(np.abs((actual - predicted) / (np.abs(actual) + epsilon))) * 100
        smape = 2.0 * np.mean(np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted) + epsilon)) * 100
        actual_direction = np.sign(actual)
        predicted_direction = np.sign(predicted)
        correct_directions = np.sum(actual_direction == predicted_direction)
        total_predictions = len(actual)
        directional_accuracy = correct_directions / total_predictions if total_predictions > 0 else 0
        tp = np.sum((actual_direction == 1) & (predicted_direction == 1))
        tn = np.sum((actual_direction == -1) & (predicted_direction == -1))
        fp = np.sum((actual_direction == -1) & (predicted_direction == 1))
        fn = np.sum((actual_direction == 1) & (predicted_direction == -1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score_val = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        correlation = np.corrcoef(actual, predicted)[0, 1] if len(actual) > 1 else 0
        theil_u = np.sqrt(np.sum((predicted - actual) ** 2)) / np.sqrt(np.sum(actual ** 2)) if np.sum(actual ** 2) > 0 else float('inf')
        bias = np.mean(predicted - actual)
        predicted_var = np.var(predicted)
        actual_var = np.var(actual)
        variance_ratio = predicted_var / actual_var if actual_var > 0 else float('inf')
        actual_ranks = pd.Series(actual).rank()
        predicted_ranks = pd.Series(predicted).rank()
        ic = np.corrcoef(actual_ranks, predicted_ranks)[0, 1] if len(actual) > 1 else 0
        return {
            'prediction_count': len(group),
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'smape': smape,
            'r2_score': r2_score_val,
            'correlation': correlation,
            'bias': bias,
            'variance_ratio': variance_ratio,
            'theil_u': theil_u,
            'information_coefficient': ic,
            'directional_accuracy': directional_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score_val,
            'specificity': specificity,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'mean_actual_return': float(np.mean(actual)),
            'mean_predicted_return': float(np.mean(predicted)),
            'std_actual_return': float(np.std(actual)),
            'std_predicted_return': float(np.std(predicted)),
        }
    if 'ForecastHorizon' in merged_df.columns:
        logger.info("Detected multi-day predictions, aggregating for evaluation...")
        aggregated_df = merged_df.groupby(['ForecastOriginDate', 'Ticker']).agg({
            'PredictedReturn': 'mean',
            'ActualReturn': 'first'
        }).reset_index()
        aggregated_df['PredictionDate'] = aggregated_df['ForecastOriginDate']
        merged_df = aggregated_df[['Ticker', 'PredictionDate', 'PredictedReturn', 'ActualReturn']]
    overall_metrics = calculate_metrics_for_group(merged_df)
    per_ticker_metrics = {}
    for ticker, group in merged_df.groupby('Ticker'):
        per_ticker_metrics[ticker] = calculate_metrics_for_group(group)
    return overall_metrics, per_ticker_metrics

def _filter_strategy_params(strategy_class, params):
    expected_params = set()
    current_class = strategy_class
    while current_class and issubclass(current_class, bt.Strategy):
        if hasattr(current_class, 'params'):
            param_keys = []
            try:
                param_keys = list(current_class.params._getkeys())
            except Exception:
                pass
            expected_params.update(param_keys)
        if current_class.__bases__:
            for base in current_class.__bases__:
                if issubclass(base, bt.Strategy):
                    current_class = base
                    break
            else:
                break
        else:
            break
    filtered_params = {k: v for k, v in params.items() if k in expected_params}
    removed_params = [k for k in params if k not in expected_params]
    logger.info(f"Final filtered parameters: {list(filtered_params.keys())}")
    if removed_params:
        logger.info(f"Removed parameters: {removed_params}")
    return filtered_params

def run_portfolio_backtest(
    run_id: str,  # This parameter is required
    strategy_class: Type['bt.Strategy'], # Use a forward reference or string
    strategy_params: Dict,
    ohlcv_data_map: Dict[str, pd.DataFrame],
    initial_cash: float = 100000.0,
    commission: float = 0.001,
    fromdate: datetime = None,
    todate: datetime = None
) -> Dict:
    """A robust function to run a Backtrader simulation for any portfolio strategy."""
    try:


        # Check if we have enough data feeds
        valid_data_feeds = {ticker: df for ticker, df in ohlcv_data_map.items() if not df.empty}
        
        if len(valid_data_feeds) < 2:
            logger.error(f"Insufficient data feeds for backtest: {len(valid_data_feeds)} feeds available, need at least 2")
            return {
                "error": f"Insufficient data feeds: {len(valid_data_feeds)} available, need at least 2",
                "FinalPortfolioValue": initial_cash,
                "InitialPortfolioValue": initial_cash,
                "TotalReturnPct": 0.0,
                "AnnualizedReturnPct": 0.0
            }

        logger.info(f"Starting backtest with {len(valid_data_feeds)} data feeds")
        logger.info(f"Strategy: {strategy_class.__name__}")
        logger.info(f"Date range: {fromdate} to {todate}")
        logger.info(f"Initial cash: ${initial_cash:,.2f}")

        cerebro = bt.Cerebro(stdstats=False)

        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=commission)
        cerebro.broker.set_shortcash(False)
        
        # Add only valid data feeds
        for ticker, df in valid_data_feeds.items():
            # The data feed itself should span the entire range (warm-up + trading)
            data_feed = bt.feeds.PandasData(dataname=df.sort_index())
            cerebro.adddata(data_feed, name=ticker)
            logger.info(f"Added data feed for {ticker}: {len(df)} bars from {df.index.min()} to {df.index.max()}")

        # Filter strategy parameters to only include expected ones
        filtered_params = _filter_strategy_params(strategy_class, strategy_params)
        
        # Log parameters without large objects
        log_params = {}
        for key, value in filtered_params.items():
            if key in ['predictions_df', 'daily_predictions']:
                if isinstance(value, pd.DataFrame):
                    log_params[key] = f"DataFrame with {len(value)} rows x {len(value.columns)} columns"
                elif isinstance(value, dict):
                    log_params[key] = f"Dictionary with {len(value)} date keys"
                else:
                    log_params[key] = f"Object of type {type(value).__name__}"
            else:
                log_params[key] = value
        
        logger.info(f"Strategy parameters after filtering: {log_params}")
        
        cerebro.addstrategy(strategy_class, **filtered_params)
        
        for analyzer_class, kwargs in get_backtrader_analyzers():
            cerebro.addanalyzer(analyzer_class, **kwargs)
            
        logger.info(f"Starting portfolio backtest with strategy: {strategy_class.__name__}...")
        logger.info(f"Starting backtest run from {fromdate} to {todate}...")
        logger.info(f"Using {len(valid_data_feeds)} data feeds: {list(valid_data_feeds.keys())}")

        results = cerebro.run(fromdate=fromdate, todate=todate)
        
        if not results:
            logger.error("Backtest returned no results!")
            return {
                "error": "Backtest returned no results",
                "FinalPortfolioValue": initial_cash,
                "InitialPortfolioValue": initial_cash,
                "TotalReturnPct": 0.0,
                "AnnualizedReturnPct": 0.0
            }
        
        strategy_instance = results[0]
        final_value = cerebro.broker.getvalue()
        
        logger.info(f"Backtest finished successfully!")
        logger.info(f"Final Portfolio Value: ${final_value:,.2f}")
        logger.info(f"Total Return: {((final_value - initial_cash) / initial_cash) * 100:.2f}%")
        
        # Check if any trades were made
        if hasattr(strategy_instance.analyzers, 'trades'):
            trades_analysis = strategy_instance.analyzers.trades.get_analysis()
            total_trades = trades_analysis.get('total', {}).get('total', 0)
            logger.info(f"Total trades executed: {total_trades}")
            
            if total_trades == 0:
                logger.warning("No trades were executed during the backtest!")
                logger.warning("This could indicate an issue with the strategy logic or parameters")
        else:
            logger.warning("Trade analyzer not available - cannot check trade count")

        rebalance_days = strategy_params.get('rebalance_days', 2) # Default to 2 if not found

        return extract_backtrader_metrics(
                    strategy_instance=results[0], 
                    run_id=run_id, 
                    backtest_start_date=fromdate,
                    rebalance_days=rebalance_days
                ) if results else {}
    except Exception as e:
        logger.error(f"Portfolio backtest simulation failed: {e}", exc_info=True)
        return {
            "error": str(e),
            "FinalPortfolioValue": initial_cash,
            "InitialPortfolioValue": initial_cash,
            "TotalReturnPct": 0.0,
            "AnnualizedReturnPct": 0.0
        }

# --- PREDICTION EVALUATION & ORCHESTRATION ---


def evaluate_and_backtest(
    run_id: str,
    predictions_df: pd.DataFrame,
    data_manager: DataManager,
    ohlcv_table_name: str,
    backtest_start_date: str,
    backtest_end_date: str,
    tickers_for_bt: List[str],
    portfolio_strategy_class: Type['bt.Strategy'], # Use a forward reference
    portfolio_strategy_params: Dict
) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """
    ### FINAL, CORRECTED VERSION ###
    This version has a clean, explicit data fetching logic that respects the
    needs of both historic and predictive strategies.
    """
    from app.backtesting.backtest import PredictedPortfolioStrategy

    merged_df = pd.DataFrame()

    bt_start_dt = pd.to_datetime(backtest_start_date, utc=True)
    bt_end_dt = pd.to_datetime(backtest_end_date, utc=True)

    # --- Step 1: Prediction Evaluation (if applicable) ---
    is_predictive = issubclass(portfolio_strategy_class, PredictedPortfolioStrategy)
    if is_predictive and not predictions_df.empty:
        logger.info("Evaluating prediction accuracy...")
        unique_tickers = sorted(list(pd.unique(predictions_df['Ticker'])))
        
        # Handle multi-day predictions
        has_multi_day_predictions = 'ForecastHorizon' in predictions_df.columns
        if has_multi_day_predictions:
            # Use origin dates for fetching actual data
            min_pred_date = pd.to_datetime(predictions_df['ForecastOriginDate'], utc=True).min()
            max_pred_date = pd.to_datetime(predictions_df['PredictionDate'], utc=True).max()
            logger.info(f"Multi-day prediction date range: {min_pred_date.strftime('%Y-%m-%d')} to {max_pred_date.strftime('%Y-%m-%d')}")
        else:
            # Single-day predictions
            min_pred_date = pd.to_datetime(predictions_df['PredictionDate'], utc=True).min()
            max_pred_date = pd.to_datetime(predictions_df['PredictionDate'], utc=True).max()
            logger.info(f"Single-day prediction date range: {min_pred_date.strftime('%Y-%m-%d')} to {max_pred_date.strftime('%Y-%m-%d')}")
        
        # Check if prediction dates align with backtest period
        if min_pred_date > bt_start_dt:
            logger.warning(f"Predictions start ({min_pred_date.strftime('%Y-%m-%d')}) after backtest start ({bt_start_dt.strftime('%Y-%m-%d')})")
        if max_pred_date < bt_end_dt:
            logger.warning(f"Predictions end ({max_pred_date.strftime('%Y-%m-%d')}) before backtest end ({bt_end_dt.strftime('%Y-%m-%d')})")
        
        fetch_start = min_pred_date - pd.Timedelta(days=1)
        fetch_end = max_pred_date + pd.Timedelta(days=60)
        actuals_data_map = data_manager.get_data_from_db(unique_tickers, fetch_start.strftime('%Y-%m-%d'), fetch_end.strftime('%Y-%m-%d'), ohlcv_table_name)
        
        all_actuals_dfs = []
        for ticker, df_ohlcv in actuals_data_map.items():
            if df_ohlcv is not None and not df_ohlcv.empty:
                df_temp = df_ohlcv[['close']].copy()
                df_temp['ActualReturn'] = df_temp['close'].pct_change().shift(-1)
                df_temp['Ticker'] = ticker
                all_actuals_dfs.append(df_temp.reset_index().rename(columns={'timestamp': 'PredictionDate'}))
        
        if all_actuals_dfs:
            actuals_df = pd.concat(all_actuals_dfs, ignore_index=True)
            actuals_df['PredictionDate'] = pd.to_datetime(actuals_df['PredictionDate'], utc=True).dt.normalize()
            actuals_df.dropna(subset=['ActualReturn'], inplace=True)
            
            # Prepare the predictions dataframe
            preds_df_prepared = predictions_df.copy()
            
            if has_multi_day_predictions:
                # For multi-day predictions, we need to match actual returns to the prediction dates
                # Each prediction origin date should be matched with the actual return on the next business day
                preds_df_prepared['PredictionDate'] = pd.to_datetime(preds_df_prepared['PredictionDate'], utc=True).dt.normalize()
                preds_df_prepared['ForecastOriginDate'] = pd.to_datetime(preds_df_prepared['ForecastOriginDate'], utc=True).dt.normalize()
                
                # For evaluation, we'll use the origin date as the key for matching
                # The actual return should be for the next business day after the origin date
                merged_df = pd.merge(
                    left=preds_df_prepared,
                    right=actuals_df[['Ticker', 'PredictionDate', 'ActualReturn']],
                    left_on=['Ticker', 'ForecastOriginDate'],
                    right_on=['Ticker', 'PredictionDate'],
                    how='inner',
                    suffixes=('_pred', '_actual')
                )
                
                # Clean up the merged dataframe
                merged_df = merged_df.rename(columns={'PredictionDate_actual': 'ActualDate'})
                merged_df = merged_df[['Ticker', 'ForecastOriginDate', 'PredictionDate_pred', 'ActualDate', 
                                     'PredictedReturn', 'ActualReturn', 'ForecastHorizon']]
                
                logger.info(f"Multi-day prediction evaluation: {len(merged_df)} matched predictions")
                
    else:
        logger.info("Historic strategy or no predictions provided. Skipping prediction evaluation.")

  # ========================= THE CORE FIX IS HERE =========================
    # --- Step 2: Determine Correct Data Fetch Range with Explicit Logic ---
    
    fetch_from_date_str: str
    
    lookback = portfolio_strategy_params.get('lookback_period', 252)
    fetch_from_dt = bt_start_dt - pd.tseries.offsets.BDay(lookback + 5) # Add a small buffer
    fetch_from_date_str = fetch_from_dt.strftime('%Y-%m-%d')
    logger.info(f"Predictive strategy: fetching data from {fetch_from_date_str} (for lookback).")

    logger.info(f"Historic strategy: fetching data from training start date {fetch_from_date_str}.")

    # The end date is always the backtest end date, plus a small buffer for safety.
    fetch_to_date_str = (bt_end_dt + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
    # ======================================================================

    logger.info(f"Final data fetch range: {fetch_from_date_str} to {fetch_to_date_str}")
    
    # --- Step 3: Fetch Data ---
    ohlcv_data_map = data_manager.get_data_from_db(
        tickers_for_bt, 
        fetch_from_date_str,
        fetch_to_date_str,
        ohlcv_table_name
    )

       # --- Step 4: Pre-process Predictions from merged_df and Add to Strategy Parameters ---
    
    if is_predictive:
        if not predictions_df.empty:
            logger.info("Pre-processing predictions into daily lookup dictionary for the strategy...")
            df = predictions_df.copy()
            # The date the forecast was MADE is the key for the decision.
            date_col = 'ForecastOriginDate' if 'ForecastOriginDate' in df.columns else 'PredictionDate'
            df['DecisionDate'] = pd.to_datetime(df[date_col]).dt.date
            
            daily_predictions_dict = {}
            grouped = df.groupby('DecisionDate')
            for date, group in grouped:
                # For each decision date, create a dictionary of {Ticker: PredictedReturn}
                daily_predictions_dict[date] = dict(zip(group['Ticker'], group['PredictedReturn']))

            # Add the READY-TO-USE dictionary to the strategy's parameters.
            portfolio_strategy_params['daily_predictions'] = daily_predictions_dict
            logger.info(f"Attached {len(daily_predictions_dict)} days of pre-processed predictions.")
        else:
            portfolio_strategy_params['daily_predictions'] = {}
            


    backtest_metrics = run_portfolio_backtest(
        run_id=run_id,
        strategy_class=portfolio_strategy_class,
        strategy_params=portfolio_strategy_params,
        ohlcv_data_map=ohlcv_data_map,
        fromdate=bt_start_dt,
        todate=bt_end_dt
    )
    


    overall_prediction_metrics, per_ticker_metrics = _calculate_prediction_metrics(merged_df)

    # --- Step 6: Combine All Metrics ---
    overall_metrics = {
        "predictive_performance": overall_prediction_metrics,
        "portfolio_performance": backtest_metrics,
        "per_ticker_predictive_performance": per_ticker_metrics
    }

    return merged_df, overall_metrics
