# trading_platform/app/backtesting/backtest.py
import backtrader as bt
import pandas as pd
import numpy as np
import logging
import warnings
from abc import abstractmethod
from pypfopt import expected_returns, risk_models, objective_functions, EfficientSemivariance, EfficientCVaR
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.exceptions import OptimizationError
from typing import Dict, Optional, Tuple, List
from scipy.optimize import minimize
from scipy.stats import norm
import cvxpy as cp
from abc import ABC, abstractmethod
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.optimize._slsqp_py')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.optimize._trustregion_constr')
logger = logging.getLogger("app.backtesting.backtest")
for handler in logging.root.handlers:
    handler.setLevel(logging.DEBUG)


#  BASE CLASS

class PortfolioStrategy(bt.Strategy):
    params = (
        ('rebalance_days', 5),
        ('portfolio_strategy', 'BaseStrategy'),
        ('enable_stop_loss_take_profit', False),
        ('stop_loss_pct', 0.05),
        ('take_profit_pct', 0.15),
        ('use_trailing_stop', False),
        ('trailing_stop_pct', 0.03),
    )
    def __init__(self):
        self.rebalance_timer = 0
        self.trading_started = False
        self.position_entry_prices = {}
        self.position_high_prices = {}
        self.stop_loss_orders = {}
        self.take_profit_orders = {}
    def log(self, txt, dt=None, level='info'):
        dt = dt or self.datas[0].datetime.date(0)
        strat_name = self.p.portfolio_strategy or self.__class__.__name__
        getattr(logger, level)(f"[{dt.isoformat()}] {strat_name}: {txt}")
    def start(self):
        self.trading_started = True
        self.log("--- Backtest Trading Period Started ---")
    def notify_trade(self, trade):
        if trade.isclosed:
            ticker = trade.data._name
            self.log(f"TRADE: {ticker} | PnL: {trade.pnl:.2f} | Size: {trade.size} | Price: {trade.price:.2f}")
            if ticker in self.position_entry_prices:
                del self.position_entry_prices[ticker]
            if ticker in self.position_high_prices:
                del self.position_high_prices[ticker]
            if ticker in self.stop_loss_orders:
                del self.stop_loss_orders[ticker]
            if ticker in self.take_profit_orders:
                del self.take_profit_orders[ticker]
    def _check_stop_loss_take_profit(self):
        if not self.p.enable_stop_loss_take_profit:
            return
        current_date = self.datas[0].datetime.date(0)
        for data in self.datas:
            ticker = data._name
            position = self.getposition(data)
            if position.size == 0:
                continue
            current_price = data.close[0]
            if ticker not in self.position_entry_prices:
                self.position_entry_prices[ticker] = current_price
                self.position_high_prices[ticker] = current_price
                self.log(f"New position in {ticker} at {current_price:.2f}")
                continue
            entry_price = self.position_entry_prices[ticker]
            if current_price > self.position_high_prices[ticker]:
                self.position_high_prices[ticker] = current_price
            price_change_pct = (current_price - entry_price) / entry_price
            high_price_change_pct = (current_price - self.position_high_prices[ticker]) / self.position_high_prices[ticker]
            stop_loss_triggered = False
            if self.p.use_trailing_stop:
                if high_price_change_pct < -self.p.trailing_stop_pct:
                    stop_loss_triggered = True
                    self.log(f"Trailing stop triggered for {ticker}: {high_price_change_pct:.1%} below high")
            else:
                if price_change_pct < -self.p.stop_loss_pct:
                    stop_loss_triggered = True
                    self.log(f"Stop-loss triggered for {ticker}: {price_change_pct:.1%} below entry")
            take_profit_triggered = price_change_pct > self.p.take_profit_pct
            if stop_loss_triggered:
                self.log(f"Executing stop-loss for {ticker} at {current_price:.2f}")
                self.order_target_percent(data=data, target=0.0)
                if ticker in self.position_entry_prices:
                    del self.position_entry_prices[ticker]
                if ticker in self.position_high_prices:
                    del self.position_high_prices[ticker]
            elif take_profit_triggered:
                self.log(f"Executing take-profit for {ticker} at {current_price:.2f}")
                self.order_target_percent(data=data, target=0.0)
                if ticker in self.position_entry_prices:
                    del self.position_entry_prices[ticker]
                if ticker in self.position_high_prices:
                    del self.position_high_prices[ticker]
    def _get_optimizer_settings(self, num_assets: int) -> tuple:
        if self.p.fully_invested:
            min_w = self.p.min_position_size
            max_w = self.p.max_position_size
            total_min_required = min_w * num_assets
            if total_min_required > 1.0:
                min_w = 1.0 / num_assets
                self.log(f"Warning: min_position_size adjusted from {self.p.min_position_size:.1%} to {min_w:.1%} for feasibility")
            if max_w < 1.0 / num_assets:
                max_w = 1.0 / num_assets
                self.log(f"Warning: max_position_size adjusted from {self.p.max_position_size:.1%} to {max_w:.1%} for feasibility")
            self.log(f"Fully Invested: min={min_w:.1%}, max={max_w:.1%} for {num_assets} assets")
        else:
            min_w, max_w = 0.0, self.p.max_position_size
            self.log(f"Cash Allowed: min={min_w:.1%}, max={max_w:.1%} for {num_assets} assets")
        weight_bounds = (min_w, max_w)
        return weight_bounds
    def _get_relaxed_optimizer_settings(self, num_assets: int) -> tuple:
        if self.p.fully_invested:
            min_w = 0.01
            max_w = min(0.30, 1.0 / max(2, num_assets // 3))
            self.log(f"Using relaxed bounds for optimization: min={min_w:.1%}, max={max_w:.1%}")
        else:
            min_w = 0.0
            max_w = 0.30
            self.log(f"Using relaxed bounds (cash allowed): min={min_w:.1%}, max={max_w:.1%}")
        return (min_w, max_w)
    def _smart_fallback_optimization(self, mu_predicted: pd.Series, S: pd.DataFrame, tickers: list, strategy_type: str = "markowitz") -> dict:
        self.log(f"Applying smart fallback optimization for {strategy_type} with relaxed constraints")
        relaxed_bounds = self._get_relaxed_optimizer_settings(len(tickers))
        try:
            if strategy_type == "markowitz":
                ef = EfficientFrontier(mu_predicted, S, weight_bounds=relaxed_bounds)
                if self.p.fully_invested:
                    ef.add_constraint(lambda w: w.sum() == 1)
                ef.max_sharpe()
                weights = ef.clean_weights()
            elif strategy_type == "min_semivariance":
                returns = self._clean_returns_data(self.get_historical_prices()[tickers].pct_change())
                es = EfficientSemivariance(mu_predicted, returns, weight_bounds=relaxed_bounds)
                if self.p.fully_invested:
                    es.add_constraint(lambda w: w.sum() == 1)
                es.min_semivariance()
                weights = es.clean_weights()
            elif strategy_type == "min_cvar":
                returns = self._clean_returns_data(self.get_historical_prices()[tickers].pct_change())
                ec = EfficientCVaR(mu_predicted, returns, weight_bounds=relaxed_bounds)
                if self.p.fully_invested:
                    ec.add_constraint(lambda w: cp.sum(w) == 1)
                ec.min_cvar()
                weights = ec.clean_weights()
            else:
                self.log(f"Unknown strategy type {strategy_type}, using equal weights")
                n_assets = len(tickers)
                equal_weight = (1.0 if self.p.fully_invested else 0.8) / n_assets
                return {t: equal_weight for t in tickers}
            self.log(f"Smart fallback optimization successful: {len(weights)} positions")
            return weights
        except Exception as e:
            self.log(f"Smart fallback optimization failed: {e}. Using equal weights.")
            n_assets = len(tickers)
            equal_weight = (1.0 if self.p.fully_invested else 0.8) / n_assets
            return {t: equal_weight for t in tickers}
    def _calculate_risk_matrix(self, prices: pd.DataFrame, risk_model: str = "ledoit_wolf") -> pd.DataFrame:
        if risk_model == "ledoit_wolf":
            return risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        elif risk_model == "semicovariance":
            returns = prices.pct_change()
            cleaned_returns = self._clean_returns_data(returns)
            if cleaned_returns.empty: return pd.DataFrame()
            semi_cov = risk_models.semicovariance(cleaned_returns, benchmark=0)
            return risk_models.fix_nonpositive_semidefinite(semi_cov, fix_method='spectral')
        else:
            return risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    def _clean_returns_data(self, returns: pd.DataFrame) -> pd.DataFrame:
        if returns.empty:
            return returns
        clean_returns = returns.copy()
        clean_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
        lower_bound = clean_returns.quantile(0.02)
        upper_bound = clean_returns.quantile(0.98)
        clean_returns.clip(lower=lower_bound, upper=upper_bound, axis=1, inplace=True)
        clean_returns.fillna(0, inplace=True)
        if not np.all(np.isfinite(clean_returns.values)):
            self.log("CRITICAL: Non-finite values remain after cleaning returns data!", level='error')
            return pd.DataFrame()
        return clean_returns
    def _rebalance_portfolio(self, target_weights: dict):
        if not target_weights:
            self.log("No target weights calculated, liquidating all positions.")
            for d in self.datas: self.order_target_percent(data=d, target=0.0)
            return
        self.log(f"Rebalancing with {len(target_weights)} target positions: {{ {', '.join([f'{k}: {v:.1%}' for k,v in sorted(target_weights.items(), key=lambda item: item[1], reverse=True)])} }}")
        current_positions = {d._name for d in self.datas if self.getposition(d).size != 0}
        target_stocks = set(target_weights.keys())
        for ticker in current_positions - target_stocks: self.order_target_percent(data=self.getdatabyname(ticker), target=0.0)
        for ticker, weight in target_weights.items(): self.order_target_percent(data=self.getdatabyname(ticker), target=weight)
        self.log(f"Portfolio value after rebalancing: {self.broker.getvalue():,.2f}")
    def get_historical_prices(self) -> pd.DataFrame:
        lb = self.p.lookback_period
        all_series = []
        for d in self.datas:
            if len(d) >= lb:
                dates = [bt.num2date(x) for x in d.datetime.get(size=lb)]
                prices = d.close.get(size=lb)
                s = pd.Series(prices, index=pd.to_datetime(dates), name=d._name)
                all_series.append(s)
            else:
                self.log(f"Not enough data for {d._name} (have {len(d)} bars, need {lb})", level='warning')
        if not all_series:
            return pd.DataFrame()
        df = pd.concat(all_series, axis=1)
        return df.dropna()

class HistoricPortfolioStrategy(PortfolioStrategy):
    params = (
        ('lookback_period', 252), ('top_k', 10), ('allow_shorting', False),
        ('fully_invested', True), ('max_position_size', 0.20), ('min_position_size', 0.02),
        ('entry_threshold', 0.01),
    )
    def next(self):
        if not self.trading_started:
            return
        if len(self) < self.p.lookback_period:
            return
        self._check_stop_loss_take_profit()
        if self.rebalance_timer % self.p.rebalance_days == 0:
            self._rebalance_portfolio(self.calculate_target_weights())
        self.rebalance_timer += 1
    @abstractmethod
    def calculate_target_weights(self) -> dict: raise NotImplementedError
class MarketAwareMixin:
    params = (('market_ticker', 'SPY'),)
    def get_separated_prices(self) -> Tuple[pd.DataFrame, pd.Series]:
        all_prices = self.get_historical_prices()
        if all_prices.empty: return pd.DataFrame(), pd.Series(dtype=float)
        market_ticker = self.p.market_ticker
        if market_ticker not in all_prices.columns:
            self.log(f"Market ticker '{market_ticker}' not found in data. Cannot separate.")
            return all_prices, pd.Series(dtype=float)
        return all_prices.drop(columns=[market_ticker]), all_prices[market_ticker]
class PredictedPortfolioStrategy(PortfolioStrategy):
    params = (
        ('lookback_period', 60),
        ('top_k', 10),
        ('allow_shorting', False),
        ('fully_invested', True),
        ('max_position_size', 0.20),
        ('min_position_size', 0.02),
        ('predictions_df', None),
        ('entry_threshold', 0.01),
        ('daily_predictions', None),
        ('enable_volatility_filter', True),
        ('volatility_lookback', 252),
        ('max_volatility_pct', 0.35),
        ('min_volatility_pct', 0.08),
        ('volatility_filter_method', 'percentile'),
        ('volatility_percentile', 70),
    )
    def __init__(self):
        super().__init__()
        self.daily_predictions = self.p.daily_predictions or {}
    def next(self):
        if not self.trading_started:
            return
        if len(self) < self.p.lookback_period:
            return
        self._check_stop_loss_take_profit()
        if self.rebalance_timer % self.p.rebalance_days == 0:
            current_date = self.datas[0].datetime.date(0)
            predictions = self.daily_predictions.get(current_date, {})
            if predictions:
                pred_values = list(predictions.values())
                if pred_values:
                    self.log(f"Predictions for {current_date}: {len(predictions)} tickers, "
                           f"mean={np.mean(pred_values):.4f}, "
                           f"std={np.std(pred_values):.4f}, "
                           f"min={np.min(pred_values):.4f}, "
                           f"max={np.max(pred_values):.4f}")
                target_weights = self.calculate_target_weights(predictions)
                if target_weights:
                    self._rebalance_portfolio(target_weights)
                else:
                    self.log(f"No target weights calculated for {current_date}, holding current positions")
            else:
                if self.rebalance_timer % (self.p.rebalance_days * 5) == 0:
                    self.log(f"No predictions available for {current_date}")
        self.rebalance_timer += 1
    def _validate_predictions(self, predictions: dict) -> bool:
        if not predictions:
            return False
        pred_values = list(predictions.values())
        if any(abs(p) > 10.0 for p in pred_values):
            self.log(f"Warning: Extreme predictions detected (max: {max(abs(p) for p in pred_values):.2f})")
            return False
        if any(not np.isfinite(p) for p in pred_values):
            self.log("Warning: Non-finite predictions detected")
            return False
        return True
    def _validate_optimization_feasibility(self, num_assets: int) -> bool:
        if self.p.fully_invested:
            total_min_required = self.p.min_position_size * num_assets
            if total_min_required > 1.0:
                self.log(f"Warning: Optimization infeasible - min_position_size {self.p.min_position_size:.1%} * {num_assets} assets = {total_min_required:.1%} > 100%")
                self.log(f"Recommendation: Reduce min_position_size to ≤ {1.0/num_assets:.1%} or increase top_k to ≥ {int(1.0/self.p.min_position_size)}")
                return False
            if self.p.max_position_size < 1.0 / num_assets:
                self.log(f"Warning: max_position_size {self.p.max_position_size:.1%} too restrictive for {num_assets} assets (need at least {1.0/num_assets:.1%})")
                self.log(f"Recommendation: Increase max_position_size to ≥ {1.0/num_assets:.1%}")
                return False
        return True
    def _get_constraint_recommendations(self, num_assets: int) -> str:
        recommendations = []
        if self.p.fully_invested:
            min_recommended = 1.0 / num_assets
            max_recommended = 1.0 / max(2, num_assets // 2)
            if self.p.min_position_size > min_recommended:
                recommendations.append(f"min_position_size ≤ {min_recommended:.1%}")
            if self.p.max_position_size < min_recommended:
                recommendations.append(f"max_position_size ≥ {min_recommended:.1%}")
            if recommendations:
                return f"Constraint recommendations for {num_assets} assets: {', '.join(recommendations)}"
        return ""
    def _apply_volatility_filter(self, predictions: dict) -> dict:
        if not self.p.enable_volatility_filter:
            return predictions
        self.log(f"Applying volatility filter (method: {self.p.volatility_filter_method})")
        prices = self.get_historical_prices()
        if prices.empty:
            self.log("No price data available for volatility filtering, skipping")
            return predictions
        returns = prices.pct_change().dropna()
        if returns.empty:
            self.log("No returns data available for volatility filtering, skipping")
            return predictions
        volatility = returns.std() * np.sqrt(252)
        if self.p.volatility_filter_method == 'percentile':
            threshold = volatility.quantile(self.p.volatility_percentile / 100)
            low_vol_stocks = volatility[volatility <= threshold].index.tolist()
            self.log(f"Volatility filter: {len(low_vol_stocks)} stocks below {threshold:.1%} volatility (bottom {self.p.volatility_percentile}%)")
        elif self.p.volatility_filter_method == 'absolute':
            low_vol_stocks = volatility[
                (volatility >= self.p.min_volatility_pct) & 
                (volatility <= self.p.max_volatility_pct)
            ].index.tolist()
            self.log(f"Volatility filter: {len(low_vol_stocks)} stocks with volatility between {self.p.min_volatility_pct:.1%} and {self.p.max_volatility_pct:.1%}")
        elif self.p.volatility_filter_method == 'market_relative':
            market_vol = volatility.mean()
            low_vol_stocks = volatility[volatility <= market_vol].index.tolist()
            self.log(f"Volatility filter: {len(low_vol_stocks)} stocks below market volatility {market_vol:.1%}")
        else:
            self.log(f"Unknown volatility filter method: {self.p.volatility_filter_method}, skipping")
            return predictions
        filtered_predictions = {
            ticker: pred for ticker, pred in predictions.items() 
            if ticker in low_vol_stocks
        }
        self.log(f"Volatility filtering: {len(predictions)} → {len(filtered_predictions)} stocks")
        if filtered_predictions:
            filtered_vols = volatility[list(filtered_predictions.keys())]
            self.log(f"Filtered stocks volatility: mean={filtered_vols.mean():.1%}, std={filtered_vols.std():.1%}, range=[{filtered_vols.min():.1%}, {filtered_vols.max():.1%}]")
        return filtered_predictions
    @abstractmethod
    def calculate_target_weights(self, todays_predictions: dict) -> dict:
        raise NotImplementedError
class RobustPredictedPortfolioStrategy(PredictedPortfolioStrategy):
    def _clean_returns_data(self, returns: pd.DataFrame) -> pd.DataFrame:
        if returns.empty: return returns
        returns.replace([np.inf, -np.inf], np.nan, inplace=True)
        lower, upper = returns.quantile(0.02), returns.quantile(0.98)
        return returns.clip(lower=lower, upper=upper, axis=1).fillna(0)


class MarkowitzHistoric(HistoricPortfolioStrategy):
    def calculate_target_weights(self) -> dict:
        prices = self.get_historical_prices()
        if len(prices.columns) < 2: return {}

        mu = expected_returns.ema_historical_return(prices)

        strong_performers = mu[mu > self.p.entry_threshold]
        if len(strong_performers) < 2:
            self.log(f"Not enough stocks above entry threshold {self.p.entry_threshold:.1%}. Found: {len(strong_performers)}")
            return {}

        top_k_tickers = strong_performers.nlargest(self.p.top_k).index.tolist()
        if len(top_k_tickers) < 2: return {}
        
        self.log(f"Selected {len(top_k_tickers)} stocks from {len(strong_performers)} above threshold {self.p.entry_threshold:.1%}")
        
        prices_filtered = prices[top_k_tickers]
        mu_filtered = mu[top_k_tickers]
        
        S = self._calculate_risk_matrix(prices_filtered, risk_model="ledoit_wolf")
        
        tickers = list(prices_filtered.columns)
        mu_filtered = mu_filtered[tickers]
        if isinstance(S, pd.DataFrame):
            S = S.loc[tickers, tickers]
        
        weight_bounds = self._get_optimizer_settings(len(tickers))
        ef = EfficientFrontier(mu_filtered, S, weight_bounds=weight_bounds)
        if self.p.fully_invested:
            ef.add_constraint(lambda w: w.sum() == 1)

        try:
            ef.max_sharpe()
            return ef.clean_weights()
        except Exception as e:
            self.log(f"Optimization failed: {e}. Using smart fallback optimization.")
            return self._smart_fallback_optimization(mu_filtered, S, tickers, "markowitz")

class MarkowitzHistoricEfficientReturn(HistoricPortfolioStrategy):

    params = (('target_return', 0.12),)
    
    def calculate_target_weights(self) -> dict:
        self.log(f"Calculating weights for Markowitz (Efficient Return)...")
        prices = self.get_historical_prices()
        if len(prices.columns) < 2: return {}
        
        try:
            mu = expected_returns.ema_historical_return(prices)
            
            strong_performers = mu[mu > self.p.entry_threshold]
            if len(strong_performers) < 2:
                self.log(f"Not enough stocks above entry threshold {self.p.entry_threshold:.1%}. Found: {len(strong_performers)}")
                return {}
            
            top_k_tickers = strong_performers.nlargest(self.p.top_k).index.tolist()
            if len(top_k_tickers) < 2: return {}
            
            self.log(f"Selected {len(top_k_tickers)} stocks from {len(strong_performers)} above threshold {self.p.entry_threshold:.1%}")
            
            prices_filtered = prices[top_k_tickers]
            mu_filtered = mu[top_k_tickers]
            S = risk_models.CovarianceShrinkage(prices_filtered).ledoit_wolf()
            
            tickers = list(prices_filtered.columns)
            mu_filtered = mu_filtered[tickers]
            if isinstance(S, pd.DataFrame):
                S = S.loc[tickers, tickers]
            
            weight_bounds = self._get_optimizer_settings(len(tickers))
            ef = EfficientFrontier(mu_filtered, S, weight_bounds=weight_bounds)
            if self.p.fully_invested:
                ef.add_constraint(lambda w: w.sum() == 1)
                
            try:
                weights = ef.efficient_return(target_return=self.p.target_return)
            except (OptimizationError, ValueError):
                
                self.log(f"Target return {self.p.target_return:.1%} not achievable, falling back to max Sharpe.")
                ef.max_sharpe()
                weights = ef.clean_weights()    
            
            return weights
                
        except Exception as e:
            self.log(f"Optimization failed: {e}. Using smart fallback optimization.")
            tickers_to_use = prices.columns if 'top_k_tickers' not in locals() else top_k_tickers
            if len(tickers_to_use) < 1: return {}
            prices_filtered = prices[tickers_to_use]
            mu_filtered = expected_returns.ema_historical_return(prices_filtered)
            S = risk_models.CovarianceShrinkage(prices_filtered).ledoit_wolf()
            return self._smart_fallback_optimization(mu_filtered, S, tickers_to_use, "markowitz")
        
class MinSemiVarianceHistoric(HistoricPortfolioStrategy):

    params = (('top_k', 10),)
    
    def calculate_target_weights(self) -> dict:
        self.log(f"Calculating weights for Min-Semi-Variance (Historic) with top_k={self.p.top_k}...")
        prices = self.get_historical_prices()
        if len(prices.columns) < 2: return {}

        returns = prices.pct_change().dropna()
        if returns.empty: return {}
        
        k = min(self.p.top_k, len(returns.columns))
        top_k_tickers = returns.std().nsmallest(k).index.tolist()
        if len(top_k_tickers) < 2:
            self.log("Not enough tickers after screening for low volatility.")
            return {}
        
        self.log(f"Screened for {len(top_k_tickers)} lowest volatility stocks.")
        
        prices_filtered = prices[top_k_tickers]
        returns_filtered = self._clean_returns_data(returns[top_k_tickers])
        if returns_filtered.empty: return {}
        
        mu_filtered = expected_returns.mean_historical_return(prices_filtered)
        
        strong_performers = mu_filtered[mu_filtered > self.p.entry_threshold]
        if len(strong_performers) < 2:
            self.log(f"Not enough low-volatility stocks above entry threshold {self.p.entry_threshold:.1%}. Found: {len(strong_performers)}")
            return {}
        
        final_tickers = strong_performers.index.tolist()
        self.log(f"Final selection: {len(final_tickers)} stocks (low volatility + return > {self.p.entry_threshold:.1%})")
        
        prices_filtered = prices[final_tickers]
        returns_filtered = returns_filtered[final_tickers]
        mu_filtered = strong_performers
        
        tickers = list(returns_filtered.columns)
        mu_filtered = mu_filtered[tickers]
        returns_filtered = returns_filtered[tickers]
        
        weight_bounds = self._get_optimizer_settings(len(tickers))

        try:
            es = EfficientSemivariance(mu_filtered, returns_filtered, weight_bounds=weight_bounds)

            if self.p.fully_invested:
                es.add_constraint(lambda w: w.sum() == 1)

            es.min_semivariance()
            
            weights = es.clean_weights()
            return weights
            
        except Exception as e:
            self.log(f"MinSemiVarianceHistoric optimization failed: {e}. Using smart fallback optimization.")
            return self._smart_fallback_optimization(mu_filtered, returns_filtered, tickers, "min_semivariance")

class MeanCVaRHistoric(HistoricPortfolioStrategy, MarketAwareMixin):
 
    params = (('target_return', 0.10),)

    def calculate_target_weights(self) -> dict:
        self.log(f"Calculating weights for Mean-CVaR (Historic)...")
        asset_prices, market_prices_series = self.get_separated_prices()
        if asset_prices.empty or market_prices_series.empty: return {}

        try:
            market_prices_df = market_prices_series.to_frame()
            mu = expected_returns.capm_return(asset_prices, market_prices=market_prices_df)

            strong_performers = mu[mu > self.p.entry_threshold]
            if len(strong_performers) < 2:
                self.log(f"Not enough stocks above entry threshold {self.p.entry_threshold:.1%}. Found: {len(strong_performers)}")
                return {}

            top_k_tickers = strong_performers.nlargest(self.p.top_k).index.tolist()
            if len(top_k_tickers) < 2: return {}
            
            self.log(f"Selected {len(top_k_tickers)} stocks from {len(strong_performers)} above threshold {self.p.entry_threshold:.1%}")
            
            mu_filtered = mu[top_k_tickers]
            
            returns_filtered = self._clean_returns_data(asset_prices[top_k_tickers].pct_change())
            if returns_filtered.empty: return {}

            tickers = list(returns_filtered.columns)
            mu_filtered = mu_filtered[tickers]
            returns_filtered = returns_filtered[tickers]
            
            weight_bounds = self._get_optimizer_settings(len(tickers))
            ec = EfficientCVaR(mu_filtered, returns_filtered, weight_bounds=weight_bounds)

            if self.p.fully_invested:
                ec.add_constraint(lambda w: cp.sum(w) == 1)

            try:
                ec.efficient_return(target_return=self.p.target_return)
            except (OptimizationError, ValueError):
                self.log("CVaR target return not achievable — falling back to min CVaR.")
                ec.min_cvar()

            return ec.clean_weights()
        except Exception as e:
            self.log(f"MeanCVaRHistoric failed: {e}. Using smart fallback optimization.")
            tickers_to_use = asset_prices.columns if 'top_k_tickers' not in locals() else top_k_tickers
            if len(tickers_to_use) < 1: return {}
            returns_data = self._clean_returns_data(asset_prices[tickers_to_use].pct_change())
            if returns_data.empty: return {}
            mu_data = expected_returns.capm_return(asset_prices[tickers_to_use], market_prices=market_prices_series.to_frame())
            return self._smart_fallback_optimization(mu_data, returns_data, tickers_to_use, "min_cvar")



# ==============================================================================
#  PREDICTIVE STRATEGIES
# ==============================================================================

class MarkowitzPredicted(RobustPredictedPortfolioStrategy):

    def calculate_target_weights(self, todays_predictions: dict) -> dict:
        self.log("Calculating weights for TopKPredicted (Equal Weights Markowitz)...")
        if not todays_predictions:
            return {}
        
        if not self._validate_predictions(todays_predictions):
            self.log("Predictions failed validation, skipping optimization")
            return {}

        volatility_filtered_predictions = self._apply_volatility_filter(todays_predictions)
        if not volatility_filtered_predictions:
            self.log("No stocks passed volatility filter")
            return {}

        positive_signals = {t: p for t, p in volatility_filtered_predictions.items() if p > self.p.entry_threshold and not pd.isna(p)}
        if not positive_signals:
            self.log("No positive signals above threshold.")
            return {}

        sorted_preds = sorted(positive_signals.items(), key=lambda x: x[1], reverse=True)
        candidate_tickers = [ticker for ticker, _ in sorted_preds[:self.p.top_k]]
        if len(candidate_tickers) < 2:
            self.log(f"Not enough candidate tickers: {len(candidate_tickers)}")
            return {}

        prices = self.get_historical_prices()
        available_tickers = [t for t in candidate_tickers if t in prices.columns]
        if len(available_tickers) < 2:
            self.log(f"Not enough available tickers: {len(available_tickers)}")
            return {}

        mu_predicted = pd.Series({t: positive_signals[t] for t in available_tickers}) * 252
        mu_predicted = mu_predicted.clip(-1.0, 1.0)  # Cap at 100% annual return
        
        self.log(f"TopK Equal Weights Markowitz for {len(available_tickers)} tickers with returns: {mu_predicted.describe()}")

        returns = self._clean_returns_data(prices[available_tickers].pct_change())
        if returns.empty or len(returns.columns) < 2:
            self.log("Insufficient returns data after cleaning")
            return {}

        S = self._calculate_risk_matrix(prices[available_tickers])

        if not self._validate_optimization_feasibility(len(available_tickers)):
            self.log("Optimization problem infeasible with current constraints, using smart fallback optimization")
            return self._smart_fallback_optimization(mu_predicted, S, available_tickers, "markowitz")

        try:
            weight_bounds = self._get_optimizer_settings(len(available_tickers))
            ef = EfficientFrontier(mu_predicted, S, weight_bounds=weight_bounds)
            
            if self.p.fully_invested:
                ef.add_constraint(lambda w: w.sum() == 1)

            ef.max_sharpe()
            weights = ef.clean_weights()
            
            self.log(f"TopK Equal Weights Markowitz optimization successful: {len(weights)} positions")
            return weights
            
        except Exception as e:
            self.log(f"TopK Equal Weights Markowitz optimization failed: {e}. Using smart fallback optimization.")
            return self._smart_fallback_optimization(mu_predicted, S, available_tickers, "markowitz")

class MinSemiVariancePredicted(RobustPredictedPortfolioStrategy):

    def calculate_target_weights(self, todays_predictions: dict) -> dict:
        self.log("Calculating weights for Min-Semi-Variance (Predicted)...")
        if not todays_predictions: return {}
        
        if not self._validate_predictions(todays_predictions):
            self.log("Predictions failed validation, skipping optimization")
            return {}
        
        volatility_filtered_predictions = self._apply_volatility_filter(todays_predictions)
        if not volatility_filtered_predictions:
            self.log("No stocks passed volatility filter")
            return {}
        
        positive_signals = {}
        for ticker, pred in volatility_filtered_predictions.items():
            if pred > self.p.entry_threshold:
                capped_pred = min(pred, 0.5)
                positive_signals[ticker] = capped_pred
        
        if len(positive_signals) < 2: 
            self.log(f"Not enough positive signals above threshold {self.p.entry_threshold}")
            return {}
        
        sorted_preds = sorted(positive_signals.items(), key=lambda x: x[1], reverse=True)
        candidate_tickers = [ticker for ticker, _ in sorted_preds[:self.p.top_k]]
        if len(candidate_tickers) < 2: 
            self.log(f"Not enough candidate tickers: {len(candidate_tickers)}")
            return {}

        prices = self.get_historical_prices()
        available_tickers = [t for t in candidate_tickers if t in prices.columns]
        if len(available_tickers) < 2: 
            self.log(f"Not enough available tickers: {len(available_tickers)}")
            return {}
        
        returns = self._clean_returns_data(prices[available_tickers].pct_change())
        if returns.empty or len(returns.columns) < 2:
            self.log("Insufficient returns data after cleaning")
            return {}

        final_tickers = list(returns.columns)
        
        if not self._validate_optimization_feasibility(len(final_tickers)):
            self.log("Optimization problem infeasible with current constraints, using smart fallback optimization")
            mu_predicted = pd.Series({t: positive_signals[t] for t in final_tickers}, index=final_tickers) * 252
            mu_predicted = mu_predicted.clip(upper=1.0)
            return self._smart_fallback_optimization(mu_predicted, returns, final_tickers, "min_semivariance")
        
        mu_predicted = pd.Series({t: positive_signals[t] for t in final_tickers}, index=final_tickers) * 252
        
        mu_predicted = mu_predicted.clip(upper=1.0)
        
        self.log(f"MinSemiVariance optimization for {len(final_tickers)} tickers with returns: {mu_predicted.describe()}")
        
        weight_bounds = self._get_optimizer_settings(len(final_tickers))

        try:
            es = EfficientSemivariance(mu_predicted, returns, weight_bounds=weight_bounds)

            if self.p.fully_invested:
                es.add_constraint(lambda w: w.sum() == 1)

            weights = es.min_semivariance()
            weights_dict = es.clean_weights(weights)
            
            self.log(f"MinSemiVariance optimization successful: {len(weights_dict)} positions")
            return weights_dict
            
        except Exception as e:
            self.log(f"MinSemiVariance optimization failed: {e}. Using smart fallback optimization.")
            mu_predicted = pd.Series({t: positive_signals[t] for t in final_tickers}, index=final_tickers) * 252
            mu_predicted = mu_predicted.clip(upper=1.0)
            return self._smart_fallback_optimization(mu_predicted, returns, final_tickers, "min_semivariance")
        
class MinCVaRPredicted(RobustPredictedPortfolioStrategy):
    def calculate_target_weights(self, todays_predictions: dict) -> dict:
        self.log("Calculating weights for Min-CVaR (Predicted)...")
        if not todays_predictions: return {}
        
        if not self._validate_predictions(todays_predictions):
            self.log("Predictions failed validation, skipping optimization")
            return {}

        volatility_filtered_predictions = self._apply_volatility_filter(todays_predictions)
        if not volatility_filtered_predictions:
            self.log("No stocks passed volatility filter")
            return {}

        positive_signals = {}
        for ticker, pred in volatility_filtered_predictions.items():
            if pred > self.p.entry_threshold:
                capped_pred = min(pred, 0.5)
                positive_signals[ticker] = capped_pred
        
        if len(positive_signals) < 2: 
            self.log(f"Not enough positive signals above threshold {self.p.entry_threshold}")
            return {}

        sorted_preds = sorted(positive_signals.items(), key=lambda x: x[1], reverse=True)
        candidate_tickers = [ticker for ticker, _ in sorted_preds[:self.p.top_k]]
        
        prices = self.get_historical_prices()
        available_tickers = [t for t in candidate_tickers if t in prices.columns]
        if len(available_tickers) < 2: 
            self.log(f"Not enough available tickers: {len(available_tickers)}")
            return {}
        
        returns = self._clean_returns_data(prices[available_tickers].pct_change())
        if returns.empty or len(returns.columns) < 2:
            self.log("Insufficient returns data after cleaning")
            return {}

        if not self._validate_optimization_feasibility(len(returns.columns)):
            self.log("Optimization problem infeasible with current constraints, using smart fallback optimization")
            final_tickers = returns.columns.tolist()
            mu_predicted = pd.Series({t: positive_signals[t] for t in final_tickers}) * 252
            mu_predicted = mu_predicted.clip(upper=1.0)
            return self._smart_fallback_optimization(mu_predicted, returns, final_tickers, "min_cvar")
        
        final_tickers = returns.columns.tolist()
        mu_predicted = pd.Series({t: positive_signals[t] for t in final_tickers}) * 252
        
        mu_predicted = mu_predicted.clip(upper=1.0)
        
        self.log(f"MinCVaR optimization for {len(final_tickers)} tickers with returns: {mu_predicted.describe()}")
        
        try:
            weight_bounds = self._get_optimizer_settings(len(final_tickers))
            
            ec = EfficientCVaR(mu_predicted, returns, weight_bounds=weight_bounds)
            
            if self.p.fully_invested:
                ec.add_constraint(lambda w: cp.sum(w) == 1)

            ec.min_cvar()
            weights = ec.clean_weights()
            
            self.log(f"MinCVaR optimization successful: {len(weights)} positions")
            return weights
            
        except Exception as e:
            self.log(f"MinCVaR optimization failed: {e}. Using smart fallback optimization.")
            return self._smart_fallback_optimization(mu_predicted, returns, final_tickers, "min_cvar")


class TopKPredicted(PredictedPortfolioStrategy):

    def calculate_target_weights(self, todays_predictions: dict) -> dict:
        self.log("Calculating weights for TopKPredicted (Equal Weights)...")
        if not todays_predictions:
            return {}
        
        if not self._validate_predictions(todays_predictions):
            self.log("Predictions failed validation, skipping allocation")
            return {}

        volatility_filtered_predictions = self._apply_volatility_filter(todays_predictions)
        if not volatility_filtered_predictions:
            self.log("No stocks passed volatility filter")
            return {}

        positive_signals = {t: p for t, p in volatility_filtered_predictions.items() if p > self.p.entry_threshold and not pd.isna(p)}
        if not positive_signals:
            self.log("No positive signals above threshold.")
            return {}

        sorted_preds = sorted(positive_signals.items(), key=lambda x: x[1], reverse=True)
        candidate_tickers = [ticker for ticker, _ in sorted_preds[:self.p.top_k]]
        if len(candidate_tickers) < 2:
            self.log(f"Not enough candidate tickers: {len(candidate_tickers)}")
            return {}

        prices = self.get_historical_prices()
        available_tickers = [t for t in candidate_tickers if t in prices.columns]
        if len(available_tickers) < 2:
            self.log("Not enough available tickers with price data.")
            return {}

        if self.p.fully_invested:
            equal_weight = 1.0 / len(available_tickers)
        else:
            equal_weight = 0.8 / len(available_tickers)
        
        weights = {ticker: equal_weight for ticker in available_tickers}
        self.log(f"TopK Equal Weights allocation: {len(weights)} stocks, {equal_weight:.1%} each: {available_tickers}")
        return weights










class EnhancedMarkowitzPredicted(PredictedPortfolioStrategy):
    """
    Conservative prediction-focused portfolio strategy optimized for weak signals.
    Uses strict filtering, conservative blending, and minimal rebalancing to extract
    value from noisy predictions while avoiding excessive trading costs.
    """
    params = (
        ('lookback_period', 252),  # Keep for covariance estimation
        ('top_k', 8),  # Reduced for higher conviction positions
        ('allow_shorting', False),
        ('fully_invested', True),
        ('predictions_df', None),
        ('entry_threshold', 0.005),  # Higher threshold for stronger signals
        ('confidence_threshold', 0.6),  # Higher confidence requirement
        ('max_position_size', 0.20),  # Max 20% in any stock
        ('min_position_size', 0.05),  # Min 5% to avoid dust positions
        ('risk_aversion', 2.0),  # Higher risk aversion
        ('prediction_weight', 0.3),  # Conservative weight on predictions
        ('historical_weight', 0.7),  # Heavy weight on historical data
        ('volatility_lookback', 60),  # Longer lookback for stability
        ('rebalance_threshold', 0.15),  # Only rebalance if weights change > 15%
        ('use_trailing_stop', True),
        ('prediction_decay_factor', 0.95),  # Faster decay for weak signals
        ('use_shrinkage', True),
        ('prediction_cap', 0.50),  # Cap predictions at 50% annual return
        ('min_sharpe_improvement', 0.1),  # Minimum Sharpe improvement to trade
        ('max_turnover_per_rebalance', 0.3),  # Max 30% portfolio turnover
        ('rebalance_frequency', 5),  # Rebalance every 5 days
    )

    def __init__(self):
        super().__init__()
        self.previous_weights = {}
        self.prediction_confidence = {}
        self.last_rebalance_date = None
        self.position_entry_prices = {}  
        self.position_high_prices = {}  
        self.stop_loss_orders = {}       
        self.take_profit_orders = {}    
        

    def next(self):
        if not self.trading_started:
            return 

        if len(self) < self.p.lookback_period:
            return
            
        if self.rebalance_timer % self.p.rebalance_frequency == 0:
            current_date = self.datas[0].datetime.date(0)
            predictions = self.daily_predictions.get(current_date, {})
            if predictions:
                self._rebalance_portfolio(self.calculate_target_weights(predictions))
        
        self.rebalance_timer += 1        
    def get_todays_predictions(self) -> dict:
        """Get today's predictions (overrides base class to add debug logging)."""
        current_date = self.datas[0].datetime.date(0)
        predictions = self.daily_predictions.get(current_date, {})
        
        if len(self) < 10:  # Only log for first 10 bars to avoid spam
            if predictions:
                self.log(f"Found {len(predictions)} predictions for {current_date}: {list(predictions.keys())}")
                # Log sample prediction values
                sample_preds = {k: f"{v:.4f}" for k, v in list(predictions.items())[:3]}
            else:
                available_dates = sorted(self.daily_predictions.keys())
                if available_dates:
                    self.log(f"No predictions for {current_date}. Available dates: {available_dates[:5]}...{available_dates[-5:] if len(available_dates) > 10 else ''}")
 
                else:
                    self.log(f"No predictions available at all for {current_date}")
                    self.log(f"daily_predictions dict is empty: {self.daily_predictions}")
        
        return predictions

    def calculate_shrinkage_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:

        cov_returns = returns.cov() * 252
        try:
            cov_matrix = risk_models.CovarianceShrinkage(cov_returns).ledoit_wolf()
        except Exception as e:
            self.log(f"Ledoit-Wolf shrinkage failed: {e}. Falling back to sample covariance.")
            cov_matrix = cov_returns.cov() * 252
        
        return pd.DataFrame(cov_matrix * 252, index=returns.columns, columns=returns.columns)

    def filter_correlated_assets(self, tickers: list, returns: pd.DataFrame) -> list:
        if len(tickers) <= 2:
            return tickers
            
        corr_matrix = returns[tickers].corr()
        filtered_tickers = []
        
        for ticker in tickers:
            is_highly_correlated = False
            for selected_ticker in filtered_tickers:
                if abs(corr_matrix.loc[ticker, selected_ticker]) > self.p.correlation_threshold:
                    is_highly_correlated = True
                    break
            
            if not is_highly_correlated:
                filtered_tickers.append(ticker)
                
        return filtered_tickers[:self.p.top_k]  # Ensure we don't exceed top_k

    def calculate_expected_returns(self, predictions: dict, tickers: list) -> pd.Series:
   
        expected_returns = pd.Series(index=tickers, dtype=float)
        
        prices = self.get_historical_prices()
        
        for ticker in tickers:
            predicted_return = predictions.get(ticker, 0) * 252
            
            if hasattr(self.p, 'prediction_decay_factor'):
                predicted_return *= (self.p.prediction_decay_factor ** 2)
            
            if hasattr(self.p, 'prediction_cap'):
                predicted_return = np.clip(predicted_return, -self.p.prediction_cap, self.p.prediction_cap)
            
            historical_return = 0.0
            try:
                if ticker in prices.columns and len(prices[ticker]) >= 10:
                    # Use longer lookback for more stable estimates
                    lookback = min(self.p.volatility_lookback, len(prices[ticker]) - 1)
                    recent_returns = prices[ticker].pct_change().tail(lookback)
                    historical_return = recent_returns.mean() * 252
                    
                    historical_return = np.clip(historical_return, -0.5, 0.5)
                else:
                    if ticker in prices.columns and len(prices[ticker]) >= 20:
                        historical_return = expected_returns.ema_historical_return(prices[ticker].to_frame()).iloc[0] * 252
                        historical_return = np.clip(historical_return, -0.5, 0.5)
            except Exception as e:
                self.log(f"Error calculating historical return for {ticker}: {e}")
                historical_return = 0.0
            
            blended_return = (self.p.prediction_weight * predicted_return + 
                            self.p.historical_weight * historical_return)
            
            blended_return = np.clip(blended_return, -0.4, 0.4)
                
            expected_returns[ticker] = blended_return
            
        # Log blending statistics for top tickers
        if len(expected_returns) > 0:
            self._log_blending_stats(predictions, prices, expected_returns, tickers)
        
        return expected_returns

    def _log_blending_stats(self, predictions: dict, prices: pd.DataFrame, expected_returns: pd.Series, tickers: list):
     
        if not tickers or len(expected_returns) == 0:
            return
            
        top_tickers = expected_returns.nlargest(3).index
        
        self.log(f"EnhancedMarkowitz blending: {self.p.prediction_weight:.1%} prediction, {self.p.historical_weight:.1%} historical")
        
        for ticker in top_tickers:
            pred_return = predictions.get(ticker, 0) * 252
            if hasattr(self.p, 'prediction_decay_factor'):
                pred_return *= (self.p.prediction_decay_factor ** 2)
            
            hist_return = 0.0
            try:
                if ticker in prices.columns and len(prices[ticker]) >= 5:
                    lookback = min(self.p.volatility_lookback, len(prices[ticker]) - 1)
                    recent_returns = prices[ticker].pct_change().tail(lookback)
                    hist_return = recent_returns.mean() * 252
            except:
                pass
            
            blended_return = expected_returns[ticker]
            
            self.log(f"{ticker}: Pred={pred_return:.1%}, Hist={hist_return:.1%}, Blended={blended_return:.1%}")

    def optimize_portfolio(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> Optional[dict]:
        """Enhanced portfolio optimization with robust handling of numerical issues."""
        n_assets = len(expected_returns)
        
        reg_factor = 1e-6
        cov_matrix_reg = cov_matrix + reg_factor * np.eye(n_assets)
        
        if self.p.fully_invested:
            min_weight = max(0.01, 1.0 / n_assets)  # At least 1% or equal weight
            max_weight = min(0.25, 1.0 / max(2, n_assets // 2))  # Cap at 25% or reasonable max
        else:
            min_weight = 0.0  # Allow zero weights (cash)
            max_weight = 0.25  # Cap at 25% per stock
        
        constraints = []
        if self.p.fully_invested:
            constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix_reg.values, weights)))
            return -(portfolio_return / (portfolio_risk + 1e-8))
        
        if self.p.allow_shorting:
            bounds = [(-max_weight, max_weight) for _ in range(n_assets)]
        else:
            bounds = [(min_weight, max_weight) for _ in range(n_assets)]
        
        x0 = np.ones(n_assets) / n_assets
        
        optimization_methods = [
            ('SLSQP', {'maxiter': 300, 'ftol': 1e-8, 'eps': 1e-8}),
            ('trust-constr', {'maxiter': 200, 'xtol': 1e-8, 'gtol': 1e-8}),
            ('L-BFGS-B', {'maxiter': 500, 'ftol': 1e-8}),
        ]
        
        for method, options in optimization_methods:
            try:
                # Suppress warnings during optimization
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = minimize(
                        objective, x0, method=method, 
                        bounds=bounds, constraints=constraints,
                        options=options
                    )
                
                if result.success:
                    weights = result.x
                    
                    # Ensure weights are within bounds (clip if necessary)
                    weights = np.clip(weights, min_weight, max_weight)
                    
                    # Filter out tiny positions and renormalize
                    weights[weights < min_weight] = 0
                    total_weight = np.sum(weights)
                    
                    if total_weight > 0:
                        if self.p.fully_invested:
                            weights = weights / total_weight  # Renormalize to sum to 1
                        # If not fully_invested, keep weights as is (allows cash)
                        
                        # Final check: ensure we have at least 2 positions
                        non_zero_weights = weights[weights > 0]
                        if len(non_zero_weights) >= 2:
                            return dict(zip(expected_returns.index, weights))
                        else:
                            # If we end up with too few positions, try equal weight
                            continue
                    else:
                        continue
                    
            except Exception as e:
                continue
        
        if self.p.fully_invested:
            equal_weight = 1.0 / n_assets
            weights_dict = {ticker: equal_weight for ticker in expected_returns.index}
            self.log(f"Optimization failed, using equal weight allocation ({equal_weight:.1%} each)")
        else:
            # Allow some cash by using smaller equal weights
            equal_weight = 0.8 / n_assets  # 80% invested, 20% cash
            weights_dict = {ticker: equal_weight for ticker in expected_returns.index}
            self.log(f"Optimization failed, using equal weight allocation with cash ({equal_weight:.1%} each, 20% cash)")
        return weights_dict

    def should_rebalance(self, new_weights: dict, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> bool:
        """
        Conservative rebalancing check that considers both weight changes and Sharpe improvement.
        Only rebalance if the improvement is significant enough to justify trading costs.
        """
        if not self.previous_weights:
            return True
            
        # Calculate current portfolio Sharpe
        current_sharpe = self._calculate_portfolio_sharpe(self.previous_weights, expected_returns, cov_matrix)
        
        # Calculate new portfolio Sharpe
        new_sharpe = self._calculate_portfolio_sharpe(new_weights, expected_returns, cov_matrix)
        
        # Calculate weight change (turnover)
        total_change = sum(abs(new_weights.get(ticker, 0) - self.previous_weights.get(ticker, 0)) 
                          for ticker in set(list(new_weights.keys()) + list(self.previous_weights.keys())))
        
        # Check if improvement is significant enough
        sharpe_improvement = new_sharpe - current_sharpe
        min_improvement = getattr(self.p, 'min_sharpe_improvement', 0.1)
        max_turnover = getattr(self.p, 'max_turnover_per_rebalance', 0.3)
        
        # Only rebalance if:
        # 1. Sharpe improvement is above threshold, OR
        # 2. Weight change is above threshold (but not too high)
        should_rebalance = (sharpe_improvement > min_improvement or 
                           (total_change > self.p.rebalance_threshold and total_change < max_turnover))
        
        if should_rebalance:
            self.log(f"Rebalancing: Sharpe {current_sharpe:.3f} → {new_sharpe:.3f} (Δ{sharpe_improvement:+.3f}), Turnover: {total_change:.1%}")
        else:
            self.log(f"Holding: Sharpe {current_sharpe:.3f} → {new_sharpe:.3f} (Δ{sharpe_improvement:+.3f}), Turnover: {total_change:.1%} (below threshold)")
        
        return should_rebalance

    def _calculate_portfolio_sharpe(self, weights: dict, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> float:
        if not weights:
            return 0.0
        
        # Find common tickers between weights and available data
        available_tickers = expected_returns.index.intersection(cov_matrix.index)
        weight_tickers = set(weights.keys())
        common_tickers = list(weight_tickers.intersection(available_tickers))
        
        if len(common_tickers) < 2:
            self.log(f"Warning: Only {len(common_tickers)} common tickers for Sharpe calculation")
            return 0.0
        
        # Recalculate weights for common tickers only
        total_weight = sum(weights[t] for t in common_tickers)
        if total_weight == 0:
            return 0.0
        
        # Normalize weights to sum to 1 for the available tickers
        normalized_weights = {t: weights[t] / total_weight for t in common_tickers}
        
        # Convert to arrays
        weights_array = np.array([normalized_weights[t] for t in common_tickers])
        returns_array = expected_returns[common_tickers].values
        cov_array = cov_matrix.loc[common_tickers, common_tickers].values
        
        # Calculate portfolio return and risk
        portfolio_return = np.dot(weights_array, returns_array)
        portfolio_risk = np.sqrt(np.dot(weights_array, np.dot(cov_array, weights_array)))
        
        # Return Sharpe ratio (with small epsilon to avoid division by zero)
        return portfolio_return / (portfolio_risk + 1e-8) if portfolio_risk > 1e-8 else 0.0

    def calculate_target_weights(self, todays_predictions: dict) -> Optional[dict]:
   
        if not todays_predictions:
            return {}
        
        # STEP 1: Apply volatility pre-filtering
        volatility_filtered_predictions = self._apply_volatility_filter(todays_predictions)
        if not volatility_filtered_predictions:
            self.log("No stocks passed volatility filter")
            return {}
        
        # STEP 2: Strict signal filtering with higher thresholds
        strong_signals = {
            ticker: pred for ticker, pred in volatility_filtered_predictions.items() 
            if abs(pred) > self.p.entry_threshold
        }
        
        if not strong_signals:
            self.log(f"No signals above conservative threshold {self.p.entry_threshold}")
            return {}
        
        # Debug: Log signal filtering
        total_signals = len(todays_predictions)
        filtered_signals = len(strong_signals)
        self.log(f"Signal filtering: {total_signals} total → {filtered_signals} above threshold {self.p.entry_threshold}")
        
        # STEP 2: Take only top K highest conviction signals
        sorted_preds = sorted(strong_signals.items(), key=lambda x: abs(x[1]), reverse=True)
        candidate_tickers = [ticker for ticker, _ in sorted_preds[:self.p.top_k]]
        
        if len(candidate_tickers) < 2:
            self.log(f"Not enough high-conviction tickers: {len(candidate_tickers)}")
            return {}
        
        # Debug: Log top-K filtering
        self.log(f"Top-K filtering: {len(strong_signals)} signals → {len(candidate_tickers)} top tickers")
        
        # STEP 3: Get historical data and filter to available tickers
        prices = self.get_historical_prices()
        if prices.empty:
            self.log("No historical price data available")
            return {}
        
        available_tickers = [t for t in candidate_tickers if t in prices.columns]
        if len(available_tickers) < 2:
            self.log(f"Not enough available tickers: {len(available_tickers)}")
            return {}
        
        # Debug: Log data availability filtering
        missing_tickers = [t for t in candidate_tickers if t not in prices.columns]
        if missing_tickers:
            self.log(f"Missing price data for tickers: {missing_tickers}")
        
        self.log(f"Conservative optimization for {len(available_tickers)} tickers: {available_tickers}")
        
        # STEP 4: Calculate conservative expected returns (capped and filtered)
        expected_returns = self.calculate_expected_returns(todays_predictions, available_tickers)
        
        # STEP 5: Calculate stable covariance matrix
        returns = prices[available_tickers].pct_change().dropna()
        cov_returns = returns.tail(self.p.volatility_lookback) if len(returns) >= self.p.volatility_lookback else returns
        cov_matrix = self.calculate_shrinkage_covariance(cov_returns)
        
        # STEP 6: Optimize portfolio
        weights = self.optimize_portfolio(expected_returns, cov_matrix)
        
        if weights is None:
            self.log("Portfolio optimization failed")
            return {}
        
        # STEP 7: Log ticker changes for debugging
        if self.previous_weights:
            old_tickers = set(self.previous_weights.keys())
            new_tickers = set(weights.keys())
            lost_tickers = old_tickers - new_tickers
            gained_tickers = new_tickers - old_tickers
            
            if lost_tickers:
                self.log(f"Lost tickers: {list(lost_tickers)}")
            if gained_tickers:
                self.log(f"Gained tickers: {list(gained_tickers)}")
        
        # STEP 8: Conservative rebalancing decision based on Sharpe improvement
        if not self.should_rebalance(weights, expected_returns, cov_matrix):
            self.log("Holding current positions - insufficient improvement")
            return self.previous_weights
        
        # STEP 9: Update and log
        self.previous_weights = weights.copy()
        self.log_portfolio_stats(weights, expected_returns, cov_matrix)
        
        return weights

    def log_portfolio_stats(self, weights: dict, expected_returns: pd.Series, cov_matrix: pd.DataFrame):
        """Log portfolio statistics with focus on prediction-based performance."""
        weights_array = np.array(list(weights.values()))
        
        expected_return = np.dot(weights_array, expected_returns)
        expected_vol = np.sqrt(np.dot(weights_array, np.dot(cov_matrix.values, weights_array)))
        sharpe_ratio = expected_return / expected_vol if expected_vol > 0 else 0
        
        # Concentration metrics
        herfindahl_index = np.sum(weights_array ** 2)
        effective_n_stocks = 1 / herfindahl_index
        
        self.log(f"Portfolio: {len(weights)} stocks, Return={expected_return:.1%}, Vol={expected_vol:.1%}, Sharpe={sharpe_ratio:.2f}")
        
        # Log only top 3 positions
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top_positions = sorted_weights[:3]
        self.log(f"Top positions: {[(ticker, f'{weight:.1%}') for ticker, weight in top_positions]}")



class PredictiveMomentumFilter(PredictedPortfolioStrategy):
    params = (
        ('top_k', 12),
        ('entry_threshold', 0.005),
        ('momentum_lookback', 20),
        ('momentum_threshold', 0.02),
        ('max_position_size', 0.20),
        ('min_position_size', 0.05),
        ('fully_invested', True),
        ('prediction_cap', 0.50),
    )

    def __init__(self):
        super().__init__()
        self.previous_weights = {}

    def calculate_target_weights(self, todays_predictions: dict) -> dict:
        self.log("Calculating weights for PredictiveMomentumFilter...")
        if not todays_predictions:
            return {}

        if not self._validate_predictions(todays_predictions):
            self.log("Predictions failed validation, skipping optimization")
            return {}

        # 1. Volatility filter
        filtered_preds = self._apply_volatility_filter(todays_predictions)
        if not filtered_preds:
            self.log("No stocks passed volatility filter")
            return {}

        # 2. Calculate price momentum
        prices = self.get_historical_prices()
        if prices.empty:
            self.log("No price data for momentum calculation")
            return {}

        returns = prices.pct_change(self.p.momentum_lookback).dropna()
        if returns.empty:
            self.log("No returns for momentum calculation")
            return {}

        momentum = returns.iloc[-1]
        momentum_signals = {}
        for ticker, pred in filtered_preds.items():
            if ticker in momentum.index:
                price_mom = momentum[ticker]
                if price_mom > self.p.momentum_threshold:
                    # Combine prediction and momentum (weight prediction more)
                    score = abs(pred) * 0.7 + price_mom * 0.3
                    momentum_signals[ticker] = score

        if not momentum_signals:
            self.log("No stocks passed momentum filter")
            return {}

        # 3. Top K by combined score
        strong = {t: s for t, s in momentum_signals.items() if s > self.p.entry_threshold}
        if len(strong) < 2:
            self.log(f"Not enough signals above threshold {self.p.entry_threshold}")
            return {}
        
        top = sorted(strong.items(), key=lambda x: x[1], reverse=True)[:self.p.top_k]
        candidate_tickers = [t for t, _ in top]
        if len(candidate_tickers) < 2:
            self.log(f"Not enough candidate tickers: {len(candidate_tickers)}")
            return {}

        # 4. Prepare expected returns (use original predictions, not scores)
        mu_pred = pd.Series({t: filtered_preds[t] for t in candidate_tickers}) * 252
        mu_pred = mu_pred.clip(-self.p.prediction_cap, self.p.prediction_cap)

        # 5. Covariance matrix
        returns = self._clean_returns_data(prices[candidate_tickers].pct_change())
        if returns.empty or len(returns.columns) < 2:
            self.log("Insufficient returns data after cleaning")
            return {}
        
        S = self._calculate_risk_matrix(prices[candidate_tickers])

        # 6. Optimization
        if not self._validate_optimization_feasibility(len(candidate_tickers)):
            self.log("Optimization infeasible, using smart fallback")
            return self._smart_fallback_optimization(mu_pred, S, candidate_tickers, "markowitz")

        try:
            weight_bounds = self._get_optimizer_settings(len(candidate_tickers))
            ef = EfficientFrontier(mu_pred, S, weight_bounds=weight_bounds)
            if self.p.fully_invested:
                ef.add_constraint(lambda w: w.sum() == 1)
            ef.max_sharpe()
            weights = ef.clean_weights()
            self.log(f"Momentum optimization successful: {len(weights)} positions")
            return weights
        except Exception as e:
            self.log(f"Momentum optimization failed: {e}. Using smart fallback optimization.")
            return self._smart_fallback_optimization(mu_pred, S, candidate_tickers, "markowitz")
