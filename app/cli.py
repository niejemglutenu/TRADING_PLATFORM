# trading_platform/app/cli.py
import argparse
import logging
import os
import sys
import json
from datetime import datetime # Keep for potential future use
#from app.backtesting.backtest import CAPMStrategy, PredictionThresholdStrategy
# --- Determine Project Root Dynamically ---
# If cli.py is in trading_platform/app/cli.py, then PROJECT_ROOT is its parent.
try:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if not os.path.isdir(os.path.join(PROJECT_ROOT, "configs")):
        # Fallback if structure is trading_platform/src/app/cli.py
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if not os.path.isdir(os.path.join(PROJECT_ROOT, "configs")):
            raise ValueError("PROJECT_ROOT ('configs' dir not found) may not be correctly determined.")
    
    # Add project_root to Python path to allow imports like 'from app.common...'
    # This is important if you run `python app/cli.py` from project_root.
    # If running with `python -m app.cli`, Python usually handles this.
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
except Exception as e_path:
    print(f"ERROR (cli.py): Could not determine project root or set up sys.path: {e_path}")
    sys.exit(1)

# --- Absolute Imports (assuming PROJECT_ROOT is now on PYTHONPATH for 'app' package) ---
try:
    from app.common.config import AppConfig # Corrected: Use the AppConfig class
    from app.orchestration.run_system import run_system
    # DataManager is instantiated within run_system or data_access layer using AppConfig
    # from app.data_ingestion.db_manager import DataManager # Not directly needed by cli.py usually
    from app.common.constants import (
        DEFAULT_PREDICT_DAYS, FEATURE_STRATEGY_KEY_DEFAULT,
        MODEL_SCOPE_DEFAULT, MODE_DEFAULT, LOG_LEVEL_DEFAULT
    )
except ModuleNotFoundError as e:
    print(f"ERROR (cli.py): Module Not Found. Details: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Determined PROJECT_ROOT: {PROJECT_ROOT}")
    print("Ensure you run from project root (e.g., 'python -m app.cli') or that PYTHONPATH is correct.")
    sys.exit(1)
except ImportError as e:
    print(f"ERROR (cli.py): Import Error. Details: {e}")
    sys.exit(1)


#STRATEGY_MAP = {
#    "capm": CAPMStrategy,
#    "prediction_threshold": PredictionThresholdStrategy
# }

# --- Argument Parsing ---
def parse_arguments(available_feature_strategies: list[str]): # Type hint for clarity
    parser = argparse.ArgumentParser(description="CLI for Trading Platform")
    parser.add_argument("--base-config", default="app_config.yaml", 
                        help="Base config filename in 'configs/' directory (e.g., app_config.yaml).")
    parser.add_argument("--profile-config", default=None, 
                        help="Profile config filename in 'configs/' (e.g., live.yaml, backtests/my_run.yaml) to override base.")
    
    parser.add_argument("--mode", choices=["backtest", "live_predict"], help="Operating mode. Overrides config.")
    parser.add_argument("--model-scope", choices=["all_stocks_model", "single_stock_model"], help="Model training scope. Overrides config.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level. Overrides config.")

    parser.add_argument("--feature-strategy-key", type=str, choices=available_feature_strategies, help="Key identifying the FeatureEngineeringStrategy class. Overrides config.")
    parser.add_argument("--tickers-to-predict", type=str, help="Comma-separated tickers (e.g., 'AAPL,MSFT'). Overrides config.")
    parser.add_argument("--training-pool-start-date", type=str, help="Overall training data pool start date (YYYY-MM-DD). Overrides config.")
    parser.add_argument("--prediction-horizon-days", type=int, help="Number of days to predict ahead. Overrides config.")
    parser.add_argument("--force-retrain-models", action="store_true", help="Force retrain initial/live model. If set, overrides config's value to True.")

    # Backtest specific parameters
    parser.add_argument("--backtest-start-date", type=str, help="Backtest period start date (YYYY-MM-DD). Overrides config.")
    parser.add_argument("--backtest-end-date", type=str, help="Backtest period end date (YYYY-MM-DD). Overrides config.")
    parser.add_argument("--force-retrain-each-step", action="store_true", help="Force retrain at each walk-forward step in backtest. If set, overrides config's value to True.")
    #parser.add_argument(
    #    "--strategy_name",
    #    type=str,
    #    choices=list(STRATEGY_MAP.keys()),  # Dynamically sync with the dict
    #    required=True,
    #    help="Choose which strategy to backtest."
    #)
    parser.add_argument("--output-artifacts-json", action='store_true', help="Output artifact paths as JSON to stdout.")



    return parser.parse_args()

def main():
    # 1. Define available strategies (can be dynamic later if needed)
    #    For now, AppConfig initialization will happen after full arg parsing.
    #    So, for choices in argparse, we might need a temporary way to get them or hardcode.
    #    Let's hardcode for simplicity in this step, then refine if needed.
    available_strategies_fallback = [
        FEATURE_STRATEGY_KEY_DEFAULT, "PastReturnsStrategy", 
        "ReturnsVariationStrategy", "ReturnsVarCorrStrategy"
    ]

    # 2. Parse ALL command-line arguments ONCE
    # The parse_arguments function defines all possible CLI arguments
    args = parse_arguments(available_feature_strategies=available_strategies_fallback)
    #strategy_class = STRATEGY_MAP[args.strategy_name]
    

    # 3. Initialize AppConfig using parsed arguments for config file paths
    
    try:
        AppConfig.initialize(
            project_root=PROJECT_ROOT,
            base_config_filename=args.base_config, # From parsed args
            profile_config_filename=args.profile_config # From parsed args
        )
    except Exception as e_cfg_init:
        print(f"FATAL (cli.py): Failed to initialize AppConfig: {e_cfg_init}")
        sys.exit(1)

    # AppConfig is now initialized.
    
    # 4. Setup Logging (CLI --log-level > AppConfig 'logging.level' > LOG_LEVEL_DEFAULT)
    log_level_str = args.log_level or AppConfig.get('logging.level', LOG_LEVEL_DEFAULT)
    try:
        numeric_log_level = getattr(logging, log_level_str.upper())
    except AttributeError:
        print(f"Warning (cli.py): Invalid log level '{log_level_str}' from CLI/config. Defaulting to INFO.")
        numeric_log_level = logging.INFO
        
    logging.basicConfig(
        level=numeric_log_level,
        format='%(asctime)s - %(levelname)s - [%(name)s:%(filename)s:%(lineno)d] - %(funcName)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True
    )
    logger = logging.getLogger("app.cli")
    logger.info(f"Trading Platform CLI started. Project Root: {PROJECT_ROOT}")
    logger.info(f"Using Base Config: '{args.base_config}', Profile Config: '{args.profile_config}'") # Use parsed args
    logger.info(f"Effective Log Level: {logging.getLevelName(logger.getEffectiveLevel())}")
    logger.debug(f"Full parsed CLI arguments: {vars(args)}")
    logger.debug(f"AppConfig instance (top-level keys): {list(AppConfig.get_instance().keys()) if AppConfig.get_instance() else 'None'}")

    # Dynamic choices for feature_strategy_key if not hardcoded in parse_arguments
    # This is if you want choices to truly come from loaded config.
    # If parse_arguments already got choices, this is just for confirmation.
    # strategies_from_config = list(AppConfig.get("feature_engineering.strategies_config", {}).keys())
    # if args.feature_strategy_key and strategies_from_config and args.feature_strategy_key not in strategies_from_config:
    #    logger.warning(f"CLI feature_strategy_key '{args.feature_strategy_key}' not found in AppConfig's strategies. Using it anyway.")


    # 5. Resolve Effective Run Parameters (CLI > AppConfig > Code Defaults)
    def get_resolved_param(cli_arg_value, config_key_path, code_default_const):
        if cli_arg_value is not None:
            return cli_arg_value
        return AppConfig.get(config_key_path, code_default_const)

    run_kwargs = {
        "mode": get_resolved_param(args.mode, "run_settings.mode", MODE_DEFAULT),
        "model_scope": get_resolved_param(args.model_scope, "run_settings.model_scope", MODEL_SCOPE_DEFAULT),
        "feature_strategy_key": get_resolved_param(args.feature_strategy_key, "run_settings.feature_strategy_key", FEATURE_STRATEGY_KEY_DEFAULT),
        "training_pool_start_date": get_resolved_param(args.training_pool_start_date, "run_settings.training_pool_start_date", "2010-01-01"),
        "prediction_horizon": get_resolved_param(args.prediction_horizon_days, "model_settings.default_prediction_horizon_days", DEFAULT_PREDICT_DAYS),
        "force_retrain_models": args.force_retrain_models or AppConfig.get("run_settings.force_retrain_models", False),
    }

    # Handle tickers_to_predict
    tickers_cli_str = args.tickers_to_predict
    tickers_from_config = AppConfig.get("run_settings.tickers_to_predict", [])
    if tickers_cli_str:
        run_kwargs["tickers_to_predict"] = [t.strip().upper() for t in tickers_cli_str.split(',') if t.strip()]
    elif tickers_from_config and isinstance(tickers_from_config, list):
        run_kwargs["tickers_to_predict"] = [str(t).strip().upper() for t in tickers_from_config if str(t).strip()]
    else:
        logger.error("CRITICAL: No 'tickers_to_predict' specified via CLI or in AppConfig. Exiting.")
        sys.exit(1)
    if not run_kwargs["tickers_to_predict"]:
        logger.error("CRITICAL: 'tickers_to_predict' list is empty after processing. Exiting.")
        sys.exit(1)

    # Get the specific feature_config_dict for the chosen strategy
    strat_key_for_dict_lookup = run_kwargs["feature_strategy_key"]
    run_kwargs["feature_config_dict"] = AppConfig.get(f"feature_engineering.strategies_config.{strat_key_for_dict_lookup}", {})
    if not run_kwargs["feature_config_dict"] and strat_key_for_dict_lookup : # Check if key was valid but no config for it
        # Check against the initially available strategies (even if fallback)
        # This warning is more accurate if available_strategies_fallback was used.
        # If available_strategies were dynamically loaded AFTER AppConfig init, this check is better.
        if strat_key_for_dict_lookup in (list(AppConfig.get("feature_engineering.strategies_config", {}).keys()) or available_strategies_fallback):
            logger.warning(f"No specific parameters found in AppConfig for feature_strategy_key '{strat_key_for_dict_lookup}'. "
                           f"The strategy will use its internal defaults.")

    if run_kwargs["mode"] == "backtest":
        run_kwargs["backtest_start_date"] = get_resolved_param(args.backtest_start_date, "backtest_settings.start_date", None)
        run_kwargs["backtest_end_date"] = get_resolved_param(args.backtest_end_date, "backtest_settings.end_date", None)
        run_kwargs["force_retrain_each_step"] = args.force_retrain_each_step or AppConfig.get("backtest_settings.force_retrain_each_step", False)
        if not run_kwargs["backtest_start_date"] or not run_kwargs["backtest_end_date"]:
            logger.error("CRITICAL: For backtest mode, 'backtest_start_date' and 'backtest_end_date' are required. Exiting.")
            sys.exit(1)

    logger.info(f"Effective parameters for run_system call:\n{json.dumps(run_kwargs, indent=2, default=str)}")

    # 6. Execute Core System Logic
    artifacts_info = None
    try:
        artifacts_info = run_system(
            project_root_path=PROJECT_ROOT,
            **run_kwargs
        )
        logger.info("run_system completed successfully.")
    except Exception as e_run:
        logger.critical(f"Core system execution (run_system) failed: {e_run}", exc_info=True)
        sys.exit(1)

    # 7. Output Artifacts
    if args.output_artifacts_json:
        if artifacts_info and isinstance(artifacts_info, dict):
            print(f"\nTRADING_PLATFORM_ARTIFACTS_JSON_START\n{json.dumps(artifacts_info, indent=2, default=str)}\nTRADING_PLATFORM_ARTIFACTS_JSON_END")
            logger.info(f"Outputted artifacts JSON. Keys: {list(artifacts_info.keys())}")
        else:
            print(f"\nTRADING_PLATFORM_ARTIFACTS_JSON_START\n{{}}\nTRADING_PLATFORM_ARTIFACTS_JSON_END")
            logger.warning("Requested artifacts JSON, but none provided by run_system or mode did not produce them.")

    logger.info("Trading Platform CLI finished.")

if __name__ == "__main__":
    main()