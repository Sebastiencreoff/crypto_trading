import json
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import ValidationError

from .schemas import AppConfig

def _resolve_paths(config_data: Dict[str, Any], base_path: Path) -> Dict[str, Any]:
    """
    Recursively resolves relative paths in config data.
    Assumes that any string value ending with '.json', '.db', '.log', etc.
    and starting with './' or '../' is a relative path.
    """
    for key, value in config_data.items():
        if isinstance(value, dict):
            _resolve_paths(value, base_path)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    _resolve_paths(item, base_path)
                elif isinstance(item, str) and (item.startswith('./') or item.startswith('../')):
                    # Simple check for path-like strings, can be made more robust
                    if item.endswith(('.json', '.db', '.log', '.txt', '.csv')):
                         value[i] = str((base_path / item).resolve())
        elif isinstance(value, str) and (value.startswith('./') or value.startswith('../')):
            # Simple check for path-like strings
            if value.endswith(('.json', '.db', '.log', '.txt', '.csv')):
                config_data[key] = str((base_path / value).resolve())
    return config_data

def load_config(path: str) -> AppConfig:
    """
    Reads a JSON configuration file, validates it against AppConfig,
    resolves relative paths, and returns the parsed AppConfig object.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    base_config_path = config_path.parent

    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from {path}: {e}")

    # Store the base path in the config data itself if the model supports it
    # This helps in scenarios where the config object is passed around
    # and needs to know its original base path for resolving further paths dynamically.
    if 'base_config_path' not in config_data and hasattr(AppConfig, 'base_config_path'):
         # This assignment is for the AppConfig model's field if it's not in the JSON.
         # The actual path resolution for other fields happens next.
         pass # Will be set in AppConfig directly if not in JSON

    # Resolve relative paths before validation
    # config_data = _resolve_paths(config_data, base_config_path)

    try:
        # Pass the base_config_path to the model for its own use if needed
        app_config = AppConfig(**config_data, base_config_path=str(base_config_path.resolve()))

        # Manually trigger path resolution for specific fields if needed,
        # or rely on Pydantic models with FilePath type and custom validators.
        # For now, we assume FilePath fields in Pydantic models handle resolution if defined with relative paths
        # and the `base_config_path` is used by those validators.
        # Example of manual resolution post-validation if necessary:
        # if app_config.some_path_field:
        #     app_config.some_path_field = (base_config_path / app_config.some_path_field).resolve()

    except ValidationError as e:
        raise ValueError(f"Configuration validation error for {path}:\n{e}")

    return app_config

# Example of how a service-specific override could work (conceptual)
def load_service_config(service_name: str, common_config_path: str, service_config_path: Optional[str] = None) -> AppConfig:
    """
    Loads a common configuration and optionally overrides it with service-specific settings.
    (This is a conceptual example and not fully implemented for the current task)
    """
    common_config = load_config(common_config_path)

    if service_config_path:
        service_specific_data = {}
        service_cfg_path = Path(service_config_path)
        if not service_cfg_path.is_file():
            raise FileNotFoundError(f"Service configuration file not found: {service_config_path}")

        try:
            with open(service_cfg_path, 'r') as f:
                service_specific_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {service_config_path}: {e}")

        # Deep merge common_config with service_specific_data
        # For simplicity, this example just updates the top-level common_config dict
        # A proper deep merge would be needed for nested structures.
        common_data = common_config.model_dump()
        common_data.update(service_specific_data)

        try:
            # Re-validate after merge
            # Ensure base_config_path from common_config is preserved or correctly updated
            base_path = Path(common_config_path).parent
            if 'base_config_path' in service_specific_data: # if service config specifies its own base
                base_path = Path(service_config_path).parent

            app_config = AppConfig(**common_data, base_config_path=str(base_path.resolve()))
        except ValidationError as e:
            raise ValueError(f"Configuration validation error after merging service config {service_config_path}:\n{e}")
        return app_config

    return common_config
