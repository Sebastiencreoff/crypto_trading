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
    currency_pair: str
    exchange_name: str
    algo_name: str
    message: Optional[str] = None

class TaskInfo(BaseModel): # For storing task details internally
    task_id: str
    currency_pair: str
    exchange_name: str
    algo_name: str
    status: str = "initializing"
    # Potentially store a reference to the trading thread/task and stop event
    # thread: Optional[Any] = None # Cannot directly serialize thread object
    # stop_event: Optional[Any] = None # Cannot directly serialize event object
    # results_queue: Optional[Any] = None # Cannot directly serialize queue object

    class Config:
        arbitrary_types_allowed = True # To allow non-pydantic types if needed, though not for direct FastAPI response

class TaskCreateResponse(BaseModel):
    task_id: str
    message: str
    details: TaskStatusResponse

class TaskStopResponse(BaseModel):
    task_id: str
    message: str

class TaskProfitResponse(BaseModel):
    task_id: str
    profit_eur: float # Assuming profit is in EUR as per existing models

class TaskResetResponse(BaseModel):
    task_id: str
    message: str
