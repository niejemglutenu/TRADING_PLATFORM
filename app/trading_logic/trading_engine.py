import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.data.live import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame


import pandas as pd
from psycopg2 import pool
import psycopg2
import numpy as np
import matplotlib.pyplot as plt

        

STOCKS = ["AAPL", "MSFT"]#,
          # "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AVGO", "PEP", "COST",
          #  "CSCO", "ADBE", "TMUS", "TXN", "NFLX", "QCOM", "INTC", "AMD", "AMGN", "HON",
          #  "SBUX", "BKNG", "MRNA", "ADI", "MDLZ", "MU", "VRTX", "GILD", "ISRG", "ABNB",
          #  "LRCX", "REGN", "SNPS", "PANW", "ASML", "CHTR", "KLAC", "FTNT", "DXCM", "CDNS",
          #  "MNST", "ADP", "CTSH", "CRWD", "ZS", "IDXX", "AEP", "PAYX", "MAR", "MCHP",
          #  "EXC", "ODFL", "PDD", "WBD", "KDP", "ATVI", "ORLY", "LCID", "ROST", "PCAR",
          #  "XEL", "EA", "BIIB", "FAST", "DLTR", "WBA", "CTAS", "SGEN", "BKR", "ANSS",
          #  "CPRT", "TEAM", "OKTA", "NXPI", "FISV", "VRSK", "AKAM", "CDW", "CEG", "SIRI",
          #  "DDOG", "ENPH", "MTCH", "BILL", "NTES", "VRSN", "TSCO", "PTC", "GEN", "CRL",
          #  "EPAM", "ALGN", "CHKP", "SWKS", "TECH", "CSGP", "ZM", "DOCU", "SPLK"]

class Trader:
    def __init__(self, api_key, secret_key):
        self.client = TradingClient(api_key, secret_key, paper=True)
        self.buy_tickers = []
        self.all_tickers = STOCKS
        self.last.timestamp = None
        

    def account_info(self):
        account = self.client.get_account()
        return account

    def get_asset_info(self, symbols, df=None):
        """
        Grabs historical prices, calculates RSI and Bollinger Bands,
        and returns DataFrame with potential buy signals.
        """
        all_tickers = STOCKS
        df_tech = []

 
        Hist = fetch_tickers(symbols, "2023-01-01", "2023-10-01")[symbol]

        for n in [14, 30, 50, 200]:
            Hist[f"ma{n}"] = sma_indicator(Hist["Close"], window=n)
            Hist[f"rsi{n}"] = RSIIndicator(Hist["Close"], window=n).rsi()
            Hist[f"bbhi{n}"] = BollingerBands(Hist["Close"], window=n).bollinger_hband_indicator()
            Hist[f"bblo{n}"] = BollingerBands(Hist["Close"], window=n).bollinger_lband_indicator()
            df_tech_temp = Hist.iloc[-1:, -16:].reset_index(drop=True)
            df_tech_temp.insert(0, "Symbol", Ticker.ticker)
            df_tech.append(df_tech_temp)
        
        
        df_tech = pd.concat(df_tech)

        # Buy criteria: price near lower BB or RSI oversold
        buy_criteria = (
            (df_tech[["bblo14", "bblo30", "bblo50", "bblo200"]] == 1).any(axis=1)
        ) | ((df_tech[["rsi14", "rsi30", "rsi50", "rsi200"]] <= 30).any(axis=1))

        filtered_df = df_tech[buy_criteria]
        self.buy_tickers = list(filtered_df["Symbol"])
        return filtered_df

import time
import logging
import os
from app.common.config import AppConfig
# ... import your DataManager, trading logic modules, signal processor, execution manager ...

def live_trading_loop():
    logger = logging.getLogger("app.live_engine")
    logger.info("Live Trading Engine starting...")
    
    # Initialize DataManager, Broker connections etc. using AppConfig
    # data_manager = DataManager(AppConfig.get('database_settings'), ...)
    # execution_manager = ExecutionManager(AppConfig.get('broker_settings'), ...)

    run_continuously = AppConfig.get('live_engine.run_continuously', True) # Get from config
    # Or: run_continuously = os.getenv("LIVE_ENGINE_RUN_CONTINUOUSLY", "true").lower() == "true"

    while run_continuously:
        logger.info("Starting new trading cycle...")
        try:
            # 1. Fetch latest market data (e.g., for last N minutes/hours)
            # latest_data = data_manager.fetch_recent_data(...)

            # 2. Generate features
            # features = feature_engineer.generate_features(latest_data, ...)
            
            # 3. Load latest model
            # model, x_scaler, y_scaler = load_latest_model_and_scalers(...)

            # 4. Make predictions
            # predictions = prediction_pipeline_for_live_tick(...) 
            # (or use the batch prediction_pipeline if it fits your prediction frequency)

            # 5. Generate trading signals
            # signals = signal_processor.generate_signals(predictions, ...)

            # 6. Risk management & order creation
            # orders_to_place = risk_manager.manage_positions_and_generate_orders(signals, current_portfolio)
            
            # 7. Execute orders
            # execution_manager.execute_orders(orders_to_place)

            # 8. Log state, update portfolio, etc.

        except Exception as e:
            logger.error(f"Error in live trading cycle: {e}", exc_info=True)
            # Implement robust error handling (e.g., circuit breakers, notifications)

        # Determine sleep interval based on trading frequency (e.g., every minute, 5 mins, hour)
        sleep_interval_seconds = int(AppConfig.get('live_engine.cycle_interval_seconds', 300))
        logger.info(f"Trading cycle complete. Sleeping for {sleep_interval_seconds} seconds.")
        time.sleep(sleep_interval_seconds)

        # Re-check run_continuously flag if you want to be able to stop it externally
        # e.g., by changing a value in a database, a file, or via an API call if you build one
        # run_continuously = check_if_still_should_run() 
    logger.info("Live Trading Engine shutting down.")
    # data_manager.close_all_connections()
    # execution_manager.close_broker_connection()

import logging
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest # etc.
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from app.data_ingestion.db_manager import DataManager # For fetching current prices if needed

logger = logging.getLogger("app.trading_engine.execution")

def make_trading_decisions_and_execute(
    predictions_df: pd.DataFrame,
    trading_client: TradingClient,
    data_manager: DataManager,
    ohlcv_table_name: str,
    risk_params: dict,
    portfolio_params: dict
) -> dict:
    results = {"orders_placed": [], "errors": []}
    
    try:
        account_info = trading_client.get_account()
        buying_power = float(account_info.buying_power)
        equity = float(account_info.equity)
        logger.info(f"Account Info: Buying Power=${buying_power:.2f}, Equity=${equity:.2f}")

        # Get current positions
        current_positions = trading_client.get_all_positions()
        position_map = {pos.symbol: pos for pos in current_positions}

    except Exception as e:
        logger.error(f"Failed to get account info or positions: {e}")
        results["errors"].append(f"Account/Position fetch error: {e}")
        return results

    max_capital_per_trade = equity * risk_params.get("max_capital_pct_per_trade", 0.05) # e.g., 5% of equity
    
    for _, row in predictions_df.iterrows():
        ticker = row['Ticker']
        predicted_return = row['PredictedReturn']
        prediction_date = row['PredictionDate'] # Date FOR which prediction is made

        # --- Simple Signal Generation Example ---
        signal = "HOLD"
        if predicted_return > risk_params.get("buy_threshold", 0.005): # Example: buy if pred > 0.5%
            signal = "BUY"
        elif predicted_return < risk_params.get("sell_threshold", -0.002): # Example: sell if pred < -0.2%
            signal = "SELL"

        logger.info(f"Ticker: {ticker}, Pred Return: {predicted_return:.4f}, Signal: {signal} for {prediction_date.date()}")

        if signal == "HOLD":
            continue

        # --- Fetch Current Price (Needed for market/limit orders) ---
        # Ideally, you'd get a more real-time quote. For EOD batch, last close might be okay.
        # For this example, let's assume we need the last known close for order sizing.
        try:
            # Fetch data up to "today" (or last trading day) to get most recent close
            today_str = pd.Timestamp.now(tz='UTC').normalize().strftime('%Y-%m-%d')
            start_fetch_current = (pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
            
            current_data_map = data_manager.get_data_from_db(
                tickers_list=[ticker],
                start_date_str=start_fetch_current, # Fetch a few days
                end_date_str=today_str, # Up to today
                table_name=ohlcv_table_name
            )
            if not current_data_map or ticker not in current_data_map or current_data_map[ticker].empty:
                logger.warning(f"Could not get current price for {ticker}. Skipping trade.")
                results["errors"].append(f"No current price for {ticker}")
                continue
            current_price = current_data_map[ticker]['close'].iloc[-1]
        except Exception as e_price:
            logger.warning(f"Error fetching current price for {ticker}: {e_price}. Skipping trade.")
            results["errors"].append(f"Price fetch error for {ticker}: {e_price}")
            continue

        if pd.isna(current_price) or current_price <= 0:
             logger.warning(f"Invalid current price {current_price} for {ticker}. Skipping trade.")
             continue


        # --- Position Sizing & Order Logic ---
        if signal == "BUY":
            if ticker in position_map: # Already have a position
                logger.info(f"Already have position in {ticker}. Holding or add-to-position logic needed.")
                # Add logic here if you want to increase position size
                continue 
            
            qty_to_buy = int(max_capital_per_trade / current_price)
            if qty_to_buy == 0:
                logger.info(f"Calculated quantity for {ticker} is 0. Capital/Price: ${max_capital_per_trade:.2f}/${current_price:.2f}. Skipping buy.")
                continue

            if buying_power < (qty_to_buy * current_price):
                logger.warning(f"Not enough buying power for {ticker}. Have ${buying_power:.2f}, Need ~${qty_to_buy * current_price:.2f}")
                # Optionally, reduce quantity if partial fill is acceptable and within strategy
                qty_to_buy = int(buying_power * 0.95 / current_price) # Use 95% of BP for this trade
                if qty_to_buy == 0:
                    continue
            
            order_data = MarketOrderRequest(
                symbol=ticker,
                qty=qty_to_buy,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY # Or GTC, etc.
            )
            try:
                order = trading_client.submit_order(order_data=order_data)
                logger.info(f"SUBMITTED BUY order for {qty_to_buy} of {ticker}: {order.id}")
                results["orders_placed"].append({"ticker": ticker, "id": order.id, "side": "BUY", "qty": qty_to_buy, "status": order.status})
                buying_power -= (qty_to_buy * current_price) # Approximate reduction
            except Exception as e_order:
                logger.error(f"Error submitting BUY order for {ticker}: {e_order}")
                results["errors"].append(f"BUY order error {ticker}: {e_order}")

        elif signal == "SELL":
            if ticker in position_map:
                position = position_map[ticker]
                qty_to_sell = float(position.qty) # Sell entire position
                if qty_to_sell <= 0: continue

                order_data = MarketOrderRequest(
                    symbol=ticker,
                    qty=qty_to_sell,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
                try:
                    order = trading_client.submit_order(order_data=order_data)
                    logger.info(f"SUBMITTED SELL order for {qty_to_sell} of {ticker}: {order.id}")
                    results["orders_placed"].append({"ticker": ticker, "id": order.id, "side": "SELL", "qty": qty_to_sell, "status": order.status})
                except Exception as e_order:
                    logger.error(f"Error submitting SELL order for {ticker}: {e_order}")
                    results["errors"].append(f"SELL order error {ticker}: {e_order}")
            else:
                logger.info(f"SELL signal for {ticker}, but no position held. No action.")
    
    return results


if __name__ == "__main__":
    # Initialize AppConfig from cli.py or a dedicated config loader for the engine
    # PROJECT_ROOT = ...
    # AppConfig.initialize(project_root=PROJECT_ROOT, base_config_name="app_config.yaml", profile_config_name="live_engine.yaml")
    # Setup logging...
    live_trading_loop()