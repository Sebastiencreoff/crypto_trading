from pydantic import BaseModel, FilePath, HttpUrl, DirectoryPath, Field
from pydantic_settings import BaseSettings, SettingsConfigDict # For Pydantic v2 style env var loading
from typing import Optional, Dict, List

# Try to import SettingsConfigDict for Pydantic v2, fallback for v1
try:
    from pydantic_settings import SettingsConfigDict
    IS_PYDANTIC_V2 = True
except ImportError:
    from pydantic import BaseConfig
    IS_PYDANTIC_V2 = False

class DatabaseConfig(BaseModel):
    type: str = Field(default="postgresql")
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    name: str = Field(default="mydb")

    if IS_PYDANTIC_V2:
        model_config = SettingsConfigDict(env_prefix='APP_DB_', extra='ignore')
    else:
        class Config(BaseConfig):
            env_prefix = 'APP_DB_'
            extra = 'ignore' # Allow other env vars not matching the fields

class ExchangeConfig(BaseModel):
    name: str
    base_url: Optional[HttpUrl] = None
    # For exchange-specific settings not covered by the above
    extra_settings: Optional[Dict] = None

class SlackConfig(BaseModel):
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

    if IS_PYDANTIC_V2:
        model_config = SettingsConfigDict(extra='ignore')
    else:
        class Config(BaseConfig):
            extra = 'ignore'
