# trading_platform/app/common/config.py
import yaml
import os
import logging
from typing import Optional, Dict, Any, Union, List
import stat
from pathlib import Path

logger_cfg = logging.getLogger(__name__) # Use a specific logger for this module

RecursiveData = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

def _parse_typed_value(value_str: str) -> Any:
    s_val = str(value_str).lower()
    if s_val == 'true': return True
    if s_val == 'false': return False
    if s_val == 'null' or s_val == 'none': return None
    try: return int(value_str)
    except ValueError:
        try: return float(value_str)
        except ValueError: return value_str

def _substitute_env_vars_recursive(data: RecursiveData) -> RecursiveData:
    if isinstance(data, dict):
        return {k: _substitute_env_vars_recursive(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_substitute_env_vars_recursive(i) for i in data]
    elif isinstance(data, str):
        try:
            if data.startswith("${") and data.endswith("}"):
                content = data[2:-1]
                parts = content.split("|", 1) # Use | as delimiter, not :-
                env_var_name = parts[0]
                default_val_str = parts[1] if len(parts) > 1 else None
                env_val = os.getenv(env_var_name)
                if env_val is not None:
                    return _parse_typed_value(env_val)
                elif default_val_str is not None:
                    return _parse_typed_value(default_val_str)
                else:
                    # If a variable is expected but not found and has no default, it might be a critical misconfiguration.
                    error_msg = f"Mandatory environment variable '{env_var_name}' is not set (referenced in config as '{data}')."
                    logger_cfg.error(error_msg)
                    raise ValueError(error_msg) # Fail fast
            return data
        except Exception as e:
            logger_cfg.warning(f"Error during env var substitution for string '{data}': {e}")
            return data
    return data

def _deep_merge_dicts(base_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> Dict[str, Any]:
    merged = base_dict.copy()
    for key, value in override_dict.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged

class AppConfig:
    _instance: Optional[Dict[str, Any]] = None
    _project_root_path: Optional[str] = None

    @staticmethod
    def initialize(
        project_root: str,
        base_config_filename: str = "app_config.yaml", # Default base config filename
        profile_config_filename: Optional[str] = None # e.g., "live.yaml" or "backtests/myrun.yaml"
    ):
        if AppConfig._instance is not None:
            logger_cfg.info("AppConfig already initialized. Skipping re-initialization.")
            return

        AppConfig._project_root_path = project_root
        
        # Initialize matplotlib cache directory
        matplotlib_cache_dir = Path(project_root) / "data" / "matplotlib_cache"
        matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ['MPLCONFIGDIR'] = str(matplotlib_cache_dir)
        
        config_dir = os.path.join(AppConfig._project_root_path, "configs")
        
        final_config: Dict[str, Any] = {}

        # 1. Load base configuration
        base_path = os.path.join(config_dir, base_config_filename)
        if os.path.exists(base_path):
            try:
                with open(base_path, 'r', encoding='utf-8') as f:
                    base_cfg_data = yaml.safe_load(f)
                if isinstance(base_cfg_data, dict):
                    final_config = base_cfg_data
                else:
                    logger_cfg.error(f"Base config '{base_path}' did not load as a dictionary. Type: {type(base_cfg_data)}")
            except Exception as e:
                logger_cfg.error(f"Error loading base config '{base_path}': {e}", exc_info=True)
        else:
            logger_cfg.warning(f"Base config file '{base_path}' not found. Starting with an empty config.")

        # 2. Load and merge profile-specific configuration
        if profile_config_filename:
            # Allows profile to be a path like "backtests/specific_run.yaml"
            profile_path = os.path.join(config_dir, profile_config_filename)
            if os.path.exists(profile_path):
                try:
                    with open(profile_path, 'r', encoding='utf-8') as f:
                        profile_cfg_data = yaml.safe_load(f)
                    if isinstance(profile_cfg_data, dict):
                        final_config = _deep_merge_dicts(final_config, profile_cfg_data)
                        logger_cfg.info(f"Successfully loaded and merged profile config: '{profile_path}'")
                    else:
                        logger_cfg.warning(f"Profile config '{profile_path}' did not load as a dictionary.")
                except Exception as e:
                    logger_cfg.error(f"Error loading profile config '{profile_path}': {e}", exc_info=True)
            else:
                logger_cfg.warning(f"Profile config file '{profile_path}' not found.")
        
        # 3. Substitute environment variables
        substituted_config = _substitute_env_vars_recursive(final_config)
        if isinstance(substituted_config, dict) or substituted_config is None:
            AppConfig._instance = substituted_config
        else:
            logger_cfg.error("Final config after env var substitution is not a dictionary. Initialization failed.")
            raise TypeError("Config after env var substitution must be a dictionary or None.")

        logger_cfg.info(f"AppConfig initialized. Project Root: '{AppConfig._project_root_path}'. "
                        f"Base: '{base_config_filename}', Profile: '{profile_config_filename}'.")
        logger_cfg.debug(f"Final resolved config keys (top-level): {list(AppConfig._instance.keys()) if AppConfig._instance else 'None'}")

    @staticmethod
    def get_instance() -> Dict[str, Any]:
        if AppConfig._instance is None:
            # This fallback initialization is risky as project_root might not be correct
            # It's best practice to ensure initialize() is called from the main entry point
            logger_cfg.critical("AppConfig.get_instance() called before initialize(). This is not recommended.")
            raise RuntimeError("AppConfig not initialized. Call AppConfig.initialize() from your main entry point.")
        return AppConfig._instance

    @staticmethod
    def get_project_root() -> str:
        if AppConfig._project_root_path is None:
            logger_cfg.critical("AppConfig.get_project_root() called before initialize().")
            raise RuntimeError("AppConfig project_root not set. Call AppConfig.initialize() first.")
        return AppConfig._project_root_path

    @staticmethod
    def get(key_path: str, default: Any = None) -> Any:
        config = AppConfig.get_instance()
        keys = key_path.split('.')
        value = config
        try:
            for key in keys:
                if not isinstance(value, dict):
                    return default
                value = value[key] # Can raise KeyError if key not found
            # If value itself is None from config, and default is also None, this is fine.
            # If value is None from config, but a non-None default is provided, prefer the config's None.
            # However, if the key_path was not found at all, then the default is returned by the except block.
            # Let's refine: if the path resolves but the value is None, return that None unless user explicitly wants to override None with default.
            # For simplicity, current behavior is: if path resolves to None, it returns None. If path doesn't resolve, it returns default.
            return value
        except KeyError:
            return default
        except TypeError: # Value became non-subscriptable (e.g., None) during path traversal
            return default

    def update_gui_from_status_data(self, status_data: Dict[str, Any]):
        # Add more detailed status information
        if 'error' in status_data:
            self.status_label.setText(f"Error: {status_data['error']}")
            self.status_label.setStyleSheet("color: red;")
        elif 'warning' in status_data:
            self.status_label.setText(f"Warning: {status_data['warning']}")
            self.status_label.setStyleSheet("color: orange;")
        else:
            self.status_label.setText(f"Status: {status_data.get('status', 'Unknown')}")
            self.status_label.setStyleSheet("color: black;")

def ensure_directory_permissions(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        # More granular permission setting
        if os.name == 'nt':  # Windows
            import win32security
            # Set Windows-specific permissions
            pass
        else:  # Unix-like
            os.chmod(directory, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        return True
    except Exception as e:
        logger_cfg.error(f"Failed to set permissions for directory {directory}: {e}")
        return False

def _convert_container_path_to_host(container_path: str) -> Path:
    if not container_path:
        return Path()
    # Remove /opt/app or /app prefix
    for prefix in ['/opt/app/', '/app/']:
        if container_path.startswith(prefix):
            container_path = container_path[len(prefix):]
    return Path(AppConfig.get_project_root()) / container_path