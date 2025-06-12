import logging
import datetime
from typing import List, Dict, Optional, Any, Tuple, Union, TYPE_CHECKING # Import TYPE_CHECKING

import pandas as pd
import numpy as np

from alpaca.data.historical import StockHistoricalDataClient
# ... other alpaca imports ...
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import Adjustment


from alpaca.common.exceptions import APIError as AlpacaPyAPIError

import psycopg

from psycopg import sql
from psycopg.abc import Query

from psycopg_pool import ConnectionPool # Type checker sees this


import os


# --- Default Constants ---
# ... (your constants) ...
DEFAULT_DB_MIN_CONN = 1
DEFAULT_DB_MAX_CONN = 10
DEFAULT_OHLCV_TABLE_NAME = "stock_ohlcv_daily"
DEFAULT_HISTORICAL_DATA_START_DT = datetime.datetime(2018, 1, 1, tzinfo=datetime.timezone.utc)
DEFAULT_ALPACA_DATA_FEED = 'iex'

RUNTIME_DB_DATA_CACHE: Dict[str, pd.DataFrame] = {}


class DataManager:
    def __init__(self,
                 db_settings: Dict[str, Any],
                 api_config: Optional[Dict[str, Any]] = None,
                 ohlcv_table_name: str = DEFAULT_OHLCV_TABLE_NAME):
        
        # ... (api_config, alpaca client init - no changes here) ...
        self.db_settings = db_settings
        self.api_config = api_config if api_config else {}
        self.ohlcv_table_name = ohlcv_table_name

        self.alpaca_api_key = self.api_config.get('alpaca_api_key')
        self.alpaca_api_secret = self.api_config.get('alpaca_api_secret')
        self.alpaca_data_feed = self.api_config.get('data_feed', DEFAULT_ALPACA_DATA_FEED).lower()

        self.data_client: Optional[StockHistoricalDataClient] = None
        if self.alpaca_api_key and self.alpaca_api_secret:
            self.data_client = StockHistoricalDataClient(
                api_key=self.alpaca_api_key,
                secret_key=self.alpaca_api_secret
            )
            logging.info(f"DataManager: Alpaca client initialized (Feed: {self.alpaca_data_feed.upper()}).")
        else:
            logging.warning("DataManager: Initialized without Alpaca API keys. API fetching will be disabled.")


        self.conn_pool: Optional[ConnectionPool] = None # This type hint should now be okay
        self._initialize_db_pool()

    def _initialize_db_pool(self):
        if self.conn_pool:
            logging.debug("DataManager: DB connection pool already initialized.")
            return
        
        try:
            # Attempt to instantiate to see if the real or dummy is used
            # This will raise ImportError if the dummy's __init__ is called
            _ = ConnectionPool(conninfo="dummy_check_only", min_size=0,max_size=5) # type: ignore
        except (ImportError, NotImplementedError): # Catch dummy's specific error or general ImportError
             logging.critical("CRITICAL: psycopg_pool library is not installed or the dummy class was invoked. Please run 'pip install psycopg_pool'.")
             raise ImportError("psycopg_pool is required but not installed.") # Ensure consistent error

        try:
            min_s = int(self.db_settings.get('min_size', DEFAULT_DB_MIN_CONN))
            max_s = int(self.db_settings.get('max_size', DEFAULT_DB_MAX_CONN))
            user = os.environ.get("DB_USER_ENV_VAR")
            password = os.environ.get("DB_PASSWORD_ENV_VAR")
            host = os.environ.get("DB_HOST_ENV_VAR", "localhost")
            port = os.environ.get("DB_PORT_ENV_VAR")
            dbname = os.environ.get("DB_NAME_ENV_VAR")

            if not all([user, host, dbname]):
                raise ValueError("DB user, host, and dbname must be provided.")

            conninfo_parts = [f"postgresql://{user}"]
            if password: conninfo_parts.append(f":{password}")
            conninfo_parts.append(f"@{host}:{port}/{dbname}")
            conninfo = "".join(conninfo_parts)
            
            logging.info(f"DataManager: Attempting to initialize DB pool for {dbname} on {host}:{port} (User: {user}).")
            self.conn_pool = ConnectionPool(conninfo=conninfo, min_size=min_s, max_size=max_s)
            
            with self.conn_pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                    if cur.fetchone() == (1,):
                        logging.info(f"DataManager: DB pool initialized and tested successfully.")
                    else:
                        raise ConnectionError("DB pool test query failed.")
        except (Exception, psycopg.Error) as e:
            logging.error(f"DataManager: DB pool initialization failed: {e}", exc_info=True)
            if self.conn_pool:
                try: self.conn_pool.close()
                except Exception as close_err: logging.error(f"Error closing pool: {close_err}")
            self.conn_pool = None
            raise ConnectionError("Failed to initialize DB pool for DataManager.") from e
    
    # Example of one more method for context:
    def close_all_connections(self):
        if self.conn_pool:
            try:
                self.conn_pool.close()
                logging.info("DataManager: DB connection pool closed.")
            except Exception as e:
                logging.error(f"DataManager: Error closing connection pool: {e}", exc_info=True)
            finally:
                self.conn_pool = None
        else:
            logging.debug("DataManager: DB connection pool was not active or already closed.")

    def _db_execute(self, query: Query, # Using the Query type alias
                    params: Optional[Union[dict, tuple, List[tuple]]] = None, # Allow dict for named placeholders
                    fetch: str = "none", many: bool = False, commit_transaction: bool = False) -> Any:
        if not self.conn_pool:
            raise ConnectionError("DataManager: DB pool not available for query execution.")
        
        result = None
        with self.conn_pool.connection() as conn:
            with conn.cursor() as cur:
                try:
                    if many and params is not None and isinstance(params, list): # executemany expects list of tuples/dicts
                        cur.executemany(query, params)
                    else: # For execute, params can be a tuple or dict
                        cur.execute(query, params)

                    if fetch == "one": result = cur.fetchone()
                    elif fetch == "all": result = cur.fetchall()
                    elif fetch == "description": result = cur.description
                    
                    if commit_transaction: conn.commit()
                except psycopg.Error as db_err:
                    diag = db_err.diag
                    msg = diag.message_primary if diag and diag.message_primary else str(db_err)
                    logging.error(f"DataManager DB Error: {msg} for query part: {str(query)[:150]}...")
                    raise
                except Exception as e:
                    logging.error(f"DataManager Unexpected DB Error: {e} for query part: {str(query)[:150]}...")
                    raise
        return result

    # --- setup_ohlcv_table method ---
    def setup_ohlcv_table(self, table_name: Optional[str] = None):
        target_table_name_str = table_name or self.ohlcv_table_name # The string name
        target_table_identifier = sql.Identifier(target_table_name_str) # For CREATE TABLE

        create_table_query = sql.SQL("""
            CREATE TABLE IF NOT EXISTS {table} (
                timestamp TIMESTAMPTZ NOT NULL,
                ticker TEXT NOT NULL,
                open NUMERIC, high NUMERIC, low NUMERIC, close NUMERIC,
                volume BIGINT, vwap NUMERIC,
                PRIMARY KEY (timestamp, ticker)
            );
        """).format(table=target_table_identifier)

        self._db_execute(create_table_query, commit_transaction=True)
        
        logging.info(f"DataManager: Standard table structure for '{target_table_name_str}' ensured.")

        try:
            if not self.conn_pool:
                 raise ConnectionError("DataManager: DB pool not available for hypertable setup.")
            with self.conn_pool.connection() as conn_hyper:
                 conn_hyper.autocommit = True 
                 with conn_hyper.cursor() as cur_hyper:
                    hypertable_query = sql.SQL("SELECT create_hypertable({table_name_literal}, 'timestamp', if_not_exists => TRUE, migrate_data => TRUE);").format(
                        table_name_literal=sql.Literal(target_table_name_str)
                    )
                    cur_hyper.execute(hypertable_query)
            logging.info(f"DataManager: Hypertable structure for '{target_table_name_str}' ensured.")
        except psycopg.Error as hyper_err:
            pgcode = hyper_err.sqlstate if hasattr(hyper_err, 'sqlstate') else None
            # Correctly check for "successful completion" or "table already hypertable"
            if pgcode == '00000' or \
               (pgcode == psycopg.errors.lookup('42P07').sqlstate if '42P07' in psycopg.errors.__dict__ else False) or \
               "already a hypertable" in str(hyper_err).lower():
                logging.info(f"Table '{target_table_name_str.lower()}' is already a hypertable or setup process completed as expected (Code: {pgcode}).")
            else:
                # Log the actual error message for other cases
                error_message = hyper_err.diag.message_primary if hyper_err.diag and hyper_err.diag.message_primary else str(hyper_err)
                logging.warning(f"DataManager: Warning during hypertable setup for '{target_table_name_str.lower()}': {error_message} (Code: {pgcode})", exc_info=False) # Set exc_info=False if message is enough
        except Exception as e: # Catch other exceptions
            logging.error(f"DataManager: Unexpected error during hypertable setup for '{target_table_name_str.lower()}': {e}", exc_info=True)


    def get_latest_timestamp_for_symbol(self, symbol: str, table_name: Optional[str] = None) -> Optional[datetime.datetime]:
        target_table_ident = sql.Identifier(table_name or self.ohlcv_table_name)
        query = sql.SQL('SELECT MAX(timestamp) FROM {table} WHERE ticker = %s;').format(table=target_table_ident)
        result = self._db_execute(query, (symbol.upper(),), fetch="one")
        
        if result and result[0] is not None:
            ts_from_db = result[0]
            if not isinstance(ts_from_db, datetime.datetime):
                logging.warning(f"Unexpected type for timestamp from DB for {symbol}: {type(ts_from_db)}")
                return None
            if ts_from_db.tzinfo is None:
                return ts_from_db.replace(tzinfo=datetime.timezone.utc)
            return ts_from_db.astimezone(datetime.timezone.utc)
        return None

    def _parse_timeframe_to_alpaca_enum(self, timeframe_input_str: str) -> TimeFrame:
        tf_clean = timeframe_input_str.strip().lower()
        mappings = {
            "1min": TimeFrame.Minute, "minute": TimeFrame.Minute,
            "5min": TimeFrame(5, TimeFrameUnit.Minute),
            "15min": TimeFrame(15, TimeFrameUnit.Minute),
            "30min": TimeFrame(30, TimeFrameUnit.Minute),
            "1hour": TimeFrame.Hour, "hour": TimeFrame.Hour, "1h": TimeFrame.Hour,
            "1day": TimeFrame.Day, "day": TimeFrame.Day, "1d": TimeFrame.Day,
        }
        if tf_clean in mappings: return mappings[tf_clean]

        if tf_clean.endswith("min") and tf_clean[:-3].isdigit():
            return TimeFrame(int(tf_clean[:-3]), TimeFrameUnit.Minute)
        if tf_clean.endswith("h") and tf_clean[:-1].isdigit():
            return TimeFrame(int(tf_clean[:-1]), TimeFrameUnit.Hour)
        if tf_clean.endswith("d") and tf_clean[:-1].isdigit():
             return TimeFrame(int(tf_clean[:-1]), TimeFrameUnit.Day)
            
        raise ValueError(f"Unsupported Alpaca timeframe string: '{timeframe_input_str}'. Supported examples: '1Min', '15Min', '1H', '1D'.")

    def fetch_ohlcv_from_api(self, 
                               symbol: str,
                               timeframe_str: str, 
                               start_dt: datetime.datetime,
                               end_dt: datetime.datetime) -> pd.DataFrame:
        if not self.data_client:
            logging.warning(f"DataManager: Alpaca client not initialized. Cannot fetch {symbol}.")
            return pd.DataFrame()
        try:
            timeframe_obj = self._parse_timeframe_to_alpaca_enum(timeframe_str)
        except ValueError as e:
            logging.error(f"DataManager: Invalid timeframe for Alpaca API fetch: {e}")
            return pd.DataFrame()

        start_utc = start_dt.astimezone(datetime.timezone.utc) if start_dt.tzinfo else start_dt.replace(tzinfo=datetime.timezone.utc)
        end_utc = end_dt.astimezone(datetime.timezone.utc) if end_dt.tzinfo else end_dt.replace(tzinfo=datetime.timezone.utc)

        if end_utc < start_utc:
            logging.debug(f"DataManager: Alpaca fetch for {symbol}: end_dt ({end_utc}) before start_dt ({start_utc}). Skipping.")
            return pd.DataFrame()
        
        if timeframe_obj.unit == TimeFrameUnit.Day and end_utc.time() == datetime.time.min:
            pass 
        elif end_utc >= datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=15): 
            if timeframe_obj.unit != TimeFrameUnit.Day:
                 end_utc = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=1)
        
        if end_utc <= start_utc:
            logging.debug(f"DataManager: Alpaca fetch for {symbol}: adjusted end_dt ({end_utc}) not after start_dt ({start_utc}). Skipping.")
            return pd.DataFrame()

        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol.upper()], 
            timeframe=timeframe_obj,
            start=start_utc, 
            end=end_utc, 
            feed=self.alpaca_data_feed, 
            adjustment=Adjustment.SPLIT
        )
        logging.debug(f"DataManager: Alpaca request for {symbol.upper()}: {start_utc} to {end_utc}, TF: {timeframe_obj}")
        try:
            market_data_bars = self.data_client.get_stock_bars(request_params)
            
            if market_data_bars and symbol in market_data_bars[symbol.upper()] == str:
                df = market_data_bars[symbol.upper()].df
                if not df.empty:
                    if isinstance(df.index, pd.MultiIndex) and symbol.upper() in df.index.get_level_values(0):
                        df_symbol = df.loc[symbol.upper()]
                    elif df.index.name == 'timestamp': 
                        df_symbol = df
                    else: 
                        logging.warning(f"Unexpected DataFrame structure from Alpaca for {symbol.upper()}. Index: {df.index}")
                        return pd.DataFrame()
                    return df_symbol.copy()
            logging.debug(f"DataManager: No data for {symbol.upper()} in Alpaca response for period.")
            return pd.DataFrame()
        except AlpacaPyAPIError as api_err:
            logging.error(f"DataManager: Alpaca API error fetching {symbol.upper()}: {api_err}")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"DataManager: Unexpected error fetching Alpaca for {symbol.upper()}: {e}", exc_info=True)
            return pd.DataFrame()

    def _standardize_and_prepare_df_for_db(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if df.empty: return pd.DataFrame()
        df_copy = df.copy()

        if isinstance(df_copy.index, pd.DatetimeIndex):
            original_index_name = df_copy.index.name
            df_copy = df_copy.reset_index()
            current_ts_col_name = original_index_name if original_index_name else 'index'
            if current_ts_col_name != 'timestamp':
                 df_copy.rename(columns={current_ts_col_name: 'timestamp'}, inplace=True)
        elif 'timestamp' not in df_copy.columns:
            logging.error(f"DataManager: DataFrame for {symbol} lacks DatetimeIndex and 'timestamp' column.")
            return pd.DataFrame()

        df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"], errors='coerce', utc=True)
        df_copy.dropna(subset=["timestamp"], inplace=True)
        if df_copy.empty: return pd.DataFrame()

        df_copy["ticker"] = symbol.upper()
        
        expected_db_cols = ["open", "high", "low", "close", "volume", "vwap"]
        for col in expected_db_cols:
            if col not in df_copy.columns:
                df_copy[col] = np.nan if col not in ['volume'] else 0
            
            
            elif col == 'volume':
                 df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0).astype(np.int64)
            else: 
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

        db_table_columns_ordered = ["timestamp", "ticker"] + expected_db_cols
        df_final = df_copy.reindex(columns=db_table_columns_ordered)
        return df_final

    def insert_ohlcv_dataframe_to_db(self, df_to_insert: pd.DataFrame, table_name: Optional[str] = None):
        target_table_ident = sql.Identifier(table_name or self.ohlcv_table_name)
        target_table_str = str(target_table_ident)

        if df_to_insert.empty:
            logging.debug(f"DataManager: No data to insert into {target_table_str}.")
            return
        
        db_cols_for_insert_str = ["timestamp", "ticker", "open", "high", "low", "close", "volume", "vwap"]
        df_ready_for_insert = df_to_insert.reindex(columns=db_cols_for_insert_str)

        records_to_insert = []
        for record_tuple_pd in df_ready_for_insert.itertuples(index=False, name=None):
            record_list = list(record_tuple_pd)
            for i, val in enumerate(record_list):
                if pd.isna(val): 
                    record_list[i] = None
            records_to_insert.append(tuple(record_list))
        
        if not records_to_insert:
            logging.debug(f"DataManager: No valid records to insert into {target_table_str}.")
            return

        cols_sql = sql.SQL(', ').join(map(sql.Identifier, db_cols_for_insert_str))
        placeholders_sql = sql.SQL(', ').join(sql.Placeholder() * len(db_cols_for_insert_str))
        conflict_target = sql.SQL("ON CONFLICT (timestamp, ticker) DO NOTHING")
        
        insert_query = sql.SQL("INSERT INTO {table} ({cols}) VALUES ({placeholders}) {conflict};").format(
            table=target_table_ident,
            cols=cols_sql,
            placeholders=placeholders_sql,
            conflict=conflict_target
        )
        
        try:
            self._db_execute(insert_query, params=records_to_insert, many=True, commit_transaction=True)
            ticker_inserted = df_to_insert['ticker'].iloc[0] if not df_to_insert.empty and 'ticker' in df_to_insert.columns else "UNKNOWN"
            logging.info(f"DataManager: DB Upsert processed {len(records_to_insert)} row candidates for {ticker_inserted} into {target_table_str}.")
        except Exception as e:
            logging.error(f"DataManager: Error during batch insert into {target_table_str}: {e}", exc_info=True)
            raise
            
    def update_historical_data_for_symbol(self,
                                          symbol: str,
                                          timeframe_str: str,
                                          table_name: Optional[str] = None,
                                          mode: str = "update",
                                          initial_load_start_dt: Optional[datetime.datetime] = None,
                                          initial_load_end_dt: Optional[datetime.datetime] = None,
                                          default_start_for_new: Optional[datetime.datetime] = None):
        target_table = table_name or self.ohlcv_table_name
        
        fetch_start_dt: Optional[datetime.datetime] = None
        fetch_end_dt: datetime.datetime = initial_load_end_dt if mode == "initial_load" and initial_load_end_dt else datetime.datetime.now(datetime.timezone.utc)

        if mode == "initial_load":
            if not (initial_load_start_dt and initial_load_end_dt):
                logging.error(f"DataManager: Initial load for {symbol} requires initial_load_start_dt and initial_load_end_dt.")
                return
            if initial_load_start_dt >= initial_load_end_dt:
                logging.error(f"DataManager: Initial load start must be before end for {symbol}.")
                return
            fetch_start_dt = initial_load_start_dt
            logging.info(f"DataManager: Initial Load for {symbol} into '{target_table}': {fetch_start_dt.date()} to {fetch_end_dt.date()} (TF: {timeframe_str})")
        
        elif mode == "update":
            latest_ts_in_db = self.get_latest_timestamp_for_symbol(symbol, target_table)
            if latest_ts_in_db:
                try:
                    timeframe_obj_temp = self._parse_timeframe_to_alpaca_enum(timeframe_str)
                    if timeframe_obj_temp.unit == TimeFrameUnit.Day:
                        fetch_start_dt = latest_ts_in_db.replace(hour=0, minute=0, second=0, microsecond=0)
                    else:
                        fetch_start_dt = latest_ts_in_db 
                except ValueError:
                    logging.warning(f"Invalid timeframe '{timeframe_str}' for update logic, defaulting.")
                    fetch_start_dt = latest_ts_in_db.replace(hour=0, minute=0, second=0, microsecond=0) if latest_ts_in_db else (default_start_for_new or DEFAULT_HISTORICAL_DATA_START_DT)
            else: 
                fetch_start_dt = default_start_for_new or DEFAULT_HISTORICAL_DATA_START_DT
                logging.info(f"DataManager: No existing data for {symbol} in {target_table}. Update will fetch from: {fetch_start_dt.date()}")
            
            if fetch_start_dt and fetch_start_dt >= fetch_end_dt:
                latest_ts_str = latest_ts_in_db.strftime('%Y-%m-%d %H:%M:%S %Z') if latest_ts_in_db else 'N/A'
                fetch_start_str = fetch_start_dt.strftime('%Y-%m-%d %H:%M:%S %Z') if fetch_start_dt else 'N/A'
                logging.info(f"DataManager: Update for {symbol} in '{target_table}': Data likely up-to-date. Last DB: {latest_ts_str}, Next fetch start: {fetch_start_str}.")
                return
            fetch_start_str_log = fetch_start_dt.strftime('%Y-%m-%d %H:%M:%S %Z') if fetch_start_dt else 'N/A_ERROR'
            fetch_end_str_log = fetch_end_dt.strftime('%Y-%m-%d %H:%M:%S %Z')
            logging.info(f"DataManager: Update for {symbol} in '{target_table}': Fetching from {fetch_start_str_log} to {fetch_end_str_log}")
        
        else:
            logging.error(f"DataManager: Invalid mode '{mode}' for update_historical_data_for_symbol.")
            return

        if not fetch_start_dt:
            logging.error(f"DataManager: fetch_start_dt not determined for {symbol} in mode {mode}.")
            return

        raw_df_api = self.fetch_ohlcv_from_api(symbol, timeframe_str, fetch_start_dt, fetch_end_dt)
        if not raw_df_api.empty:
            prepared_df = self._standardize_and_prepare_df_for_db(raw_df_api, symbol)
            if not prepared_df.empty:
                self.insert_ohlcv_dataframe_to_db(prepared_df, target_table)
            else:
                logging.debug(f"DataManager: API data for {symbol} for {timeframe_str} became empty after standardization.")
        else:
            logging.debug(f"DataManager: No new data from API for {symbol} ({timeframe_str}) for {fetch_start_dt.date()} to {fetch_end_dt.date()}.")

    def get_data_from_db(self,
                         tickers_list: List[str],
                         start_date_str: str,
                         end_date_str: str,
                         table_name: Optional[str] = None,
                         use_runtime_cache: bool = True
                        ) -> Dict[str, pd.DataFrame]:
        target_table_ident = sql.Identifier(table_name or self.ohlcv_table_name)
        target_table_str = str(target_table_ident)
        output_map: Dict[str, pd.DataFrame] = {}
        tickers_to_fetch_from_db_set = set()

        try:
            required_start_dt = pd.to_datetime(start_date_str, utc=True)
            required_end_dt = pd.to_datetime(end_date_str, utc=True)
            if required_start_dt > required_end_dt:
                logging.warning(f"DataManager get_data_from_db: Start date {start_date_str} is after end date {end_date_str}.")
                return {t.upper(): pd.DataFrame() for t in tickers_list}
        except Exception as e:
            logging.error(f"DataManager: Invalid dates for get_data_from_db ('{start_date_str}', '{end_date_str}'): {e}")
            return {t.upper(): pd.DataFrame() for t in tickers_list}

        for ticker_raw in tickers_list:
            if not ticker_raw or not isinstance(ticker_raw, str): continue
            ticker_upper = ticker_raw.upper()
            
            if use_runtime_cache and ticker_upper in RUNTIME_DB_DATA_CACHE:
                cached_df = RUNTIME_DB_DATA_CACHE[ticker_upper]
                if not cached_df.empty and \
                   isinstance(cached_df.index, pd.DatetimeIndex) and \
                   not cached_df.index.min(skipna=True) > required_start_dt and \
                   not cached_df.index.max(skipna=True) < required_end_dt:
                    slice_df = cached_df[
                        (cached_df.index >= required_start_dt) & (cached_df.index <= required_end_dt)
                    ].copy()
                    output_map[ticker_upper] = slice_df
                    logging.debug(f"DataManager: Cache sufficient for {ticker_upper} for range {start_date_str} to {end_date_str}.")
                    continue
            tickers_to_fetch_from_db_set.add(ticker_upper)

        if tickers_to_fetch_from_db_set:
            tickers_to_fetch_list = list(tickers_to_fetch_from_db_set)
            logging.info(f"DataManager: Fetching from DB for {tickers_to_fetch_list} for table '{target_table_str}', range {start_date_str} to {end_date_str}")
            
            query_template = sql.SQL("""
                SELECT timestamp, ticker, open, high, low, close, volume, vwap
                FROM {table}
                WHERE ticker = ANY(%s) AND timestamp BETWEEN %s AND %s 
                ORDER BY ticker, timestamp ASC; 
            """).format(table=target_table_ident)
            
            all_fetched_data = self._db_execute(query_template, (tickers_to_fetch_list, required_start_dt, required_end_dt), fetch="all")
            
            if all_fetched_data:
                desc_query_template = sql.SQL("SELECT timestamp, ticker, open, high, low, close, volume, vwap FROM {table} WHERE FALSE;")
                colnames_desc_result = self._db_execute(desc_query_template.format(table=target_table_ident), fetch="description")
                colnames = [desc.name for desc in colnames_desc_result] if colnames_desc_result else []

                if colnames:
                    full_df_from_db = pd.DataFrame(all_fetched_data, columns=colnames)
                    if not full_df_from_db.empty:
                         full_df_from_db['timestamp'] = pd.to_datetime(full_df_from_db['timestamp'], utc=True)

                    for ticker_symbol_db in tickers_to_fetch_list:
                        df_for_ticker = full_df_from_db[full_df_from_db['ticker'] == ticker_symbol_db].copy() if not full_df_from_db.empty else pd.DataFrame()
                        
                        if not df_for_ticker.empty:
                            df_for_ticker.set_index('timestamp', inplace=True)
                            for col in ['open', 'high', 'low', 'close', 'volume', 'vwap']:
                                if col in df_for_ticker.columns:
                                    df_for_ticker[col] = pd.to_numeric(df_for_ticker[col], errors='coerce')
                            
                            if use_runtime_cache:
                                RUNTIME_DB_DATA_CACHE[ticker_symbol_db] = df_for_ticker.copy() # Cache the full fetched range
                            
                            output_map[ticker_symbol_db] = df_for_ticker[
                                (df_for_ticker.index >= required_start_dt) & (df_for_ticker.index <= required_end_dt)
                            ].copy() # Slice for output
                        else:
                            output_map[ticker_symbol_db] = pd.DataFrame()
                            if use_runtime_cache: RUNTIME_DB_DATA_CACHE[ticker_symbol_db] = pd.DataFrame()
                else:
                    logging.error(f"DataManager: Could not retrieve column names for DB query for table {target_table_str}.")
                    for t_sym_err in tickers_to_fetch_list: output_map[t_sym_err] = pd.DataFrame()
            else:
                for t_sym_no_data in tickers_to_fetch_list:
                     output_map[t_sym_no_data] = pd.DataFrame()
                     if use_runtime_cache: RUNTIME_DB_DATA_CACHE[t_sym_no_data] = pd.DataFrame()
        
        for ticker_original_upper in (s.upper() for s in tickers_list if s):
            if ticker_original_upper not in output_map:
                output_map[ticker_original_upper] = pd.DataFrame()
                
        return output_map

    def clear_runtime_cache(self, ticker: Optional[str] = None):
        global RUNTIME_DB_DATA_CACHE
        if ticker:
            ticker_upper = ticker.upper()
            if ticker_upper in RUNTIME_DB_DATA_CACHE:
                del RUNTIME_DB_DATA_CACHE[ticker_upper]
                logging.info(f"DataManager: Runtime cache cleared for {ticker_upper}.")
        else:
            RUNTIME_DB_DATA_CACHE.clear()
            logging.info("DataManager: Entire runtime DB data cache cleared.")


# --- Main Public Function (Wrapper for updating data) ---
def update_all_market_data(
    symbols_to_process: List[str],
    db_settings: Dict[str, Any],
    api_config: Dict[str, Any],
    ohlcv_table_name: str = DEFAULT_OHLCV_TABLE_NAME,
    timeframe_str: str = "1Day",
    mode: str = "update",
    initial_load_start_date_str: Optional[str] = None,
    initial_load_end_date_str: Optional[str] = None,
    default_start_date_for_new_str: Optional[str] = None,
    skip_api_calls: bool = False
) -> bool:
    
    effective_symbols = [s.upper() for s in symbols_to_process if s and isinstance(s, str)]
    if not effective_symbols and mode not in ["ensure_table_only"]:
        logging.info("update_all_market_data: No valid symbols provided for data processing.")
        return True

    def parse_date_str_to_dt(date_str: Optional[str], default_dt: datetime.datetime) -> datetime.datetime:
        if date_str:
            try: 
                parsed_pd_dt = pd.to_datetime(date_str)
                # Ensure datetime.datetime and UTC
                dt_obj = parsed_pd_dt.to_pydatetime()
                if dt_obj.tzinfo is None:
                    return dt_obj.replace(tzinfo=datetime.timezone.utc)
                return dt_obj.astimezone(datetime.timezone.utc)
            except Exception as e: 
                logging.warning(f"Invalid date string '{date_str}': {e}. Using default: {default_dt.strftime('%Y-%m-%d')}")
                return default_dt
        return default_dt

    now_utc_dt = datetime.datetime.now(datetime.timezone.utc)
    eff_default_start_new_dt = parse_date_str_to_dt(default_start_date_for_new_str, DEFAULT_HISTORICAL_DATA_START_DT)
    eff_init_load_start_dt = parse_date_str_to_dt(initial_load_start_date_str, eff_default_start_new_dt)
    eff_init_load_end_dt = parse_date_str_to_dt(initial_load_end_date_str, now_utc_dt)

    manager_instance: Optional[DataManager] = None
    operation_successful = False
    try:
        manager_instance = DataManager(db_settings=db_settings, api_config=api_config, ohlcv_table_name=ohlcv_table_name)
        
        logging.info(f"update_all_market_data: Ensuring table '{manager_instance.ohlcv_table_name}' (TF context: {timeframe_str}).")
        manager_instance.setup_ohlcv_table()

        if mode == "ensure_table_only":
            logging.info(f"Mode 'ensure_table_only': Table '{manager_instance.ohlcv_table_name}' structure ensured.")
            operation_successful = True
        elif skip_api_calls:
            logging.info("update_all_market_data: skip_api_calls is True. No API calls will be made.")
            operation_successful = True
        elif not manager_instance.data_client:
            logging.error(f"DataManager's Alpaca client not available for mode '{mode}'. Cannot proceed with API operations.")
            operation_successful = False # Explicitly false as API operation was intended but impossible
        elif mode == "initial_load":
            if effective_symbols:
                for symbol in effective_symbols:
                    manager_instance.update_historical_data_for_symbol(
                        symbol, timeframe_str, mode="initial_load",
                        initial_load_start_dt=eff_init_load_start_dt, 
                        initial_load_end_dt=eff_init_load_end_dt
                    )
                operation_successful = True
            else: logging.warning("Initial load mode but no symbols to process.")
        elif mode == "update":
            if effective_symbols:
                for symbol in effective_symbols:
                    manager_instance.update_historical_data_for_symbol(
                        symbol, timeframe_str, mode="update",
                        default_start_for_new=eff_default_start_new_dt
                    )
                operation_successful = True
            else: logging.warning("Update mode but no symbols to process.")
        else:
            logging.error(f"Invalid mode '{mode}'. Valid: 'update', 'initial_load', 'ensure_table_only'.")
            operation_successful = False
        
    except (ValueError, ConnectionError) as config_conn_err:
        logging.error(f"Config/Connection Error in update_all_market_data: {config_conn_err}", exc_info=False)
        operation_successful = False
    except psycopg.Error as db_err:
        logging.error(f"Database error in update_all_market_data: {db_err.diag.message_primary if db_err.diag else db_err}", exc_info=False)
        operation_successful = False
    except Exception as e:
        logging.error(f"Unexpected error in update_all_market_data: {e}", exc_info=True)
        operation_successful = False
    finally:
        if manager_instance:
            manager_instance.close_all_connections()
    return operation_successful