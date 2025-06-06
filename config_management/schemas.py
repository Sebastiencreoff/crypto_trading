from pydantic import BaseModel, FilePath, HttpUrl, DirectoryPath
from typing import Optional, Dict, List

class DatabaseConfig(BaseModel):
    type: str
    host: str
    port: int
    username: str
    password: str
    name: str

class ExchangeConfig(BaseModel):
    name: str
    api_key: str
    secret_key: str
    base_url: Optional[HttpUrl] = None
    # For exchange-specific settings not covered by the above
    extra_settings: Optional[Dict] = None

class SlackConfig(BaseModel):
    bot_token: str # Slack Bot User OAuth Token (xoxb-...)
    default_channel_id: str # Default channel ID to send messages to (e.g., C12345678)

class AlgoConfig(BaseModel):
    name: str
    # Parameters specific to the algorithm
    parameters: Dict

class AppConfig(BaseModel):
    service_name: str
    database: DatabaseConfig
    # Allow multiple exchange configurations
    exchanges: List[ExchangeConfig]
    slack: Optional[SlackConfig] = None
    # Allow multiple algorithm configurations
    algorithms: Optional[List[AlgoConfig]] = None
    # For any other top-level settings
    other_settings: Optional[Dict] = None
    # Define a base path for resolving relative paths in config
    base_config_path: Optional[DirectoryPath] = None
    # URL for the notification service
    notification_service_url: Optional[HttpUrl] = None
