from pydantic import BaseModel
from typing import Optional, Dict, Any

class CreateTaskRequest(BaseModel):
    currency_pair: str # e.g., "BTCUSDT"
    transaction_amount: float # Amount for each transaction in quote currency (e.g., USDT)
    exchange_name: str # Name of the exchange to use, e.g., "binance"
    algo_name: str # Name of the algorithm to use, e.g., "default_algo"
    # Optional: specific parameters to override algo settings for this task
    algo_override_params: Optional[Dict[str, Any]] = None

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str # e.g., "running", "stopped", "error", "initializing"
    currency_pair: Optional[str] = None
    exchange_name: Optional[str] = None
    algo_name: Optional[str] = None
    message: Optional[str] = None

class TaskCreateResponse(BaseModel):
    task_id: str
    message: str

class TaskStopResponse(BaseModel):
    task_id: str
    message: str

class TaskProfitResponse(BaseModel):
    task_id: str
    profit_eur: float # Assuming profit is in EUR as per existing models

class TaskResetResponse(BaseModel):
    task_id: str
    message: str
