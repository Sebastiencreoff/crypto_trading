import logging
import uuid # Keep uuid for now, might be removed if TaskManager handles all ID generation
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends # BackgroundTasks might be removed if not used elsewhere
from typing import Optional # Dict might be removed if active_tasks is gone

# Adjust imports based on project structure
# Assuming config_management, trading_service, crypto_trading are top-level or in PYTHONPATH
from config_management.loader import load_config # May not be needed if app_config is sole source
from config_management.schemas import AppConfig, ExchangeConfig, AlgoConfig, SlackConfig
from crypto_trading.database.core_operations import get_session as get_db_session_from_engine # Renamed to avoid conflict
from crypto_trading.config import init as init_global_config, app_config as global_app_config # For DB engine init
from crypto_trading.task_manager import TaskManager # New import
from .slack_client import SlackNotifier # Moved from notification_service

from .models import CreateTaskRequest, TaskStatusResponse, TaskCreateResponse, TaskStopResponse, TaskProfitResponse, TaskResetResponse # Removed TaskInfo

import os # Added for environment variable access

# --- Globals / App State ---
task_manager_instance: Optional[TaskManager] = None
slack_notifier_instance: Optional[SlackNotifier] = None # For Slack notifications
app_config: Optional[AppConfig] = None # Loaded config

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI App Lifespan (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global app_config, task_manager_instance, slack_notifier_instance
    logger.info("FastAPI application starting up...")
    # Load configuration
    config_file_path = os.environ.get("APP_CONFIG_FILE_PATH")
    if not config_file_path:
        logger.error("APP_CONFIG_FILE_PATH environment variable not set.")
        raise RuntimeError("APP_CONFIG_FILE_PATH environment variable not set.")
    logger.info(f"Loading configuration from: {config_file_path}")

    try:
        init_global_config(config_file_path)
        app_config = global_app_config # global_app_config is set by init_global_config

        if not app_config: # Check if global_app_config got populated
             logger.error("Failed to load application configuration during startup (app_config is None).")
             raise RuntimeError("Failed to load application configuration.")
        logger.info(f"Configuration loaded successfully for service: {app_config.service_name}")

        # Initialize TaskManager
        task_manager_instance = TaskManager()
        logger.info("TaskManager initialized.")

        # Initialize SlackNotifier if Slack config is present
        if app_config.slack:
            logger.info("Slack configuration found, initializing SlackNotifier...")
            slack_notifier_instance = SlackNotifier(slack_config=app_config.slack)
            if slack_notifier_instance.client:
                logger.info("SlackNotifier initialized successfully.")
            else:
                logger.warning("SlackNotifier initialization failed (client not available). Check SLACK_BOT_TOKEN and config.")
        else:
            logger.info("No Slack configuration found in app_config. SlackNotifier will not be initialized.")

    except Exception as e:
        logger.error(f"Critical error during application startup: {e}", exc_info=True)
        raise
    yield
    logger.info("FastAPI application shutting down...")
    # TaskManager should handle its own cleanup if necessary, or rely on K8s to terminate pods.
    # No explicit per-task cleanup here anymore.
    logger.info("Shutdown complete.")


app = FastAPI(title="Trading Service", version="0.1.0", lifespan=lifespan)

# --- Database Session Dependency (No longer directly used by endpoints in this file) ---
# The get_db_session function is removed as endpoints now rely on TaskManager
# or are not implemented. TaskManager tasks will handle their own DB sessions.
# The global DB engine is initialized in lifespan via crypto_trading.config.init_global_config.

# --- Utility Functions ---
def send_slack_notification(message: str, channel_id: Optional[str] = None) -> bool:
    """
    Sends a notification message via Slack using the global notifier instance.
    This is a synchronous wrapper suitable for use from threaded code.
    """
    if not slack_notifier_instance:
        logger.warning("Attempted to send Slack notification, but SlackNotifier is not initialized.")
        return False

    # The actual SlackNotifier.send_message is synchronous.
    success = slack_notifier_instance.send_message(message=message, channel_id=channel_id)
    if success:
        logger.info(f"Successfully sent Slack message: {message[:50]}...")
    else:
        logger.error(f"Failed to send Slack message: {message[:50]}...")
    return success

# --- API Endpoints ---

@app.post("/tasks", response_model=TaskCreateResponse, status_code=201)
async def create_trading_task(task_request: CreateTaskRequest):
    """
    Create and launch a new trading task via TaskManager.
    """
    if not app_config or not task_manager_instance:
        logger.error("Server configuration or TaskManager not loaded.")
        raise HTTPException(status_code=500, detail="Server configuration or TaskManager not loaded.")

    logger.info(f"Received request to create task for {task_request.currency_pair} on {task_request.exchange_name} with algo {task_request.algo_name}")

    # Find the selected exchange configuration
    selected_exchange: Optional[ExchangeConfig] = None
    for ex_cfg in app_config.exchanges:
        if ex_cfg.name == task_request.exchange_name:
            selected_exchange = ex_cfg
            break
    if not selected_exchange:
        logger.error(f"Exchange '{task_request.exchange_name}' not found in configuration.")
        raise HTTPException(status_code=400, detail=f"Exchange '{task_request.exchange_name}' not configured.")

    # Find the selected algorithm configuration
    selected_algo_config: Optional[AlgoConfig] = None
    if app_config.algorithms:
        for algo_cfg in app_config.algorithms:
            if algo_cfg.name == task_request.algo_name:
                selected_algo_config = algo_cfg
                break
    if not selected_algo_config:
        logger.error(f"Algorithm '{task_request.algo_name}' not found in configuration.")
        raise HTTPException(status_code=400, detail=f"Algorithm '{task_request.algo_name}' not configured.")

    # Handle algorithm parameter overrides
    effective_algo_config = selected_algo_config
    if task_request.algo_override_params:
        temp_params = selected_algo_config.parameters.copy(deep=True) # Use deepcopy if parameters are nested
        temp_params.update(task_request.algo_override_params)
        effective_algo_config = AlgoConfig(name=selected_algo_config.name, parameters=temp_params)

    # Prepare app_settings for TaskManager
    # Ensure database_url and notification_service_url are correctly obtained
    db_url = str(app_config.database.database_url) if app_config.database and app_config.database.database_url else None
    notif_url = str(app_config.notification_service_url) if app_config.notification_service_url else None

    if not db_url:
        logger.error("Database URL is not configured, cannot create task.")
        raise HTTPException(status_code=500, detail="Database URL is not configured.")

    # Create a dictionary for app_settings, excluding potentially sensitive or irrelevant fields for the task
    # For example, exclude 'exchanges' and 'algorithms' lists as they are handled separately
    app_settings_for_task = {
        "service_name": app_config.service_name,
        "database": {"database_url": db_url, "create_tables": app_config.database.create_tables}, # Pass specific DB settings
        "notification_service_url": notif_url,
        # Add other AppConfig fields that the remote task might need
        # For example, if 'other_settings' contains relevant info:
        "other_settings": app_config.other_settings.dict() if app_config.other_settings else {}
    }


    # Prepare task_parameters for TaskManager
    # Source interval from app_config.other_settings if available or use a sensible default
    default_interval = 60  # Default to 60 seconds
    interval = app_config.other_settings.get("default_algo_interval", default_interval) if app_config.other_settings else default_interval

    task_specific_params = {
        "currency_pair": task_request.currency_pair, # Changed from "currency" to "currency_pair" for clarity
        "transaction_amount": task_request.transaction_amount,
        "interval": interval
    }

    # Construct the overall config object for TaskManager
    task_config_obj = {
        "app_settings": app_settings_for_task,
        "exchange_settings": selected_exchange.dict(),
        "algo_settings": effective_algo_config.dict(),
        "task_parameters": task_specific_params,
        # TaskManager might also need info like which trading script/module to run.
        # This should be part of TaskManager's own configuration or conventions.
    }

    try:
        logger.info(f"Submitting task to TaskManager with config: {task_config_obj}")
        # The TaskManager generates its own UUID for the task, which is used for the k8s job name.
        k8s_task_id = task_manager_instance.create_task(task_config_obj)

        if k8s_task_id is None:
            logger.error("TaskManager failed to create task, returned None for k8s_task_id.")
            raise HTTPException(status_code=500, detail="Failed to create trading task: TaskManager returned no ID.")

        logger.info(f"Task successfully submitted to TaskManager. Kubernetes Task ID: {k8s_task_id}")

        # Return the k8s_task_id. The 'details' field was removed from TaskCreateResponse.
        return TaskCreateResponse(
            task_id=k8s_task_id, # This is the ID from TaskManager/K8s
            message="Trading task successfully submitted to TaskManager."
        )

    except Exception as e:
        logger.error(f"Error creating trading task via TaskManager: {e}", exc_info=True)
        # Catch specific exceptions from TaskManager if any are defined, or handle generally.
        raise HTTPException(status_code=500, detail=f"Failed to create trading task: {str(e)}")


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get the status of a specific trading task from TaskManager.
    """
    if not task_manager_instance:
        logger.error("TaskManager not available.")
        raise HTTPException(status_code=500, detail="TaskManager not available.")

    logger.info(f"Requesting status for task_id: {task_id} from TaskManager.")
    status_str = task_manager_instance.get_task_status(task_id)

    if status_str is None:
        logger.warning(f"Task {task_id} not found by TaskManager or status is None.")
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found or status unavailable.")

    # Generic message based on status
    message = f"Task is currently {status_str}."
    if status_str == "Completed": # K8s specific
        message = "Task has completed."
    elif status_str == "Failed": # K8s specific
        message = "Task has failed."
    elif status_str == "Pending": # K8s specific
        message = "Task is pending to start."
    elif status_str == "Running": # K8s specific
        message = "Task is currently running."
    elif status_str == "Unknown": # K8s specific or TaskManager internal
        message = "Task status is unknown."


    # currency_pair, exchange_name, algo_name are not stored by TaskManager.
    # They are marked as Optional in TaskStatusResponse model and will be None.
    return TaskStatusResponse(
        task_id=task_id,
        status=status_str,
        message=message
        # currency_pair, exchange_name, algo_name will default to None
    )

@app.post("/tasks/{task_id}/stop", response_model=TaskStopResponse)
async def stop_trading_task(task_id: str):
    if not task_manager_instance:
        logger.error("TaskManager not available.")
        raise HTTPException(status_code=500, detail="TaskManager not available.")

    logger.info(f"Requesting to stop task_id: {task_id} via TaskManager.")
    success = task_manager_instance.stop_task(task_id)

    if success:
        logger.info(f"Task {task_id} successfully signaled to stop by TaskManager.")
        return TaskStopResponse(task_id=task_id, message="Task signaled to stop successfully.")
    else:
        # This could mean the task was not found, or already stopped, or failed to stop.
        # TaskManager.stop_task should ideally return more specific info or log it.
        logger.warning(f"Failed to stop task {task_id} via TaskManager, or task not found/already stopped.")
        # Returning a 404 if stop_task implies not found might be reasonable.
        # For now, using a generic message.
        # Consider if TaskManager distinguishes "not found" from "failed to signal stop".
        # If stop_task returns False for "not found", a 404 is appropriate.
        # If it could also be "found but failed to signal", 500 or 400 might be better.
        # Assuming False means "not found or cannot be stopped (e.g. already completed/failed)"
        raise HTTPException(status_code=404, detail=f"Failed to stop task {task_id}. It might not exist, or may have already completed/failed.")


# TODO: Add endpoints for /tasks/{task_id}/profits and /tasks/{task_id}/reset
# These functionalities are not straightforward with TaskManager controlling remote k8s jobs.
# The concept of direct profit calculation or state reset of a remote, containerized task
# needs a different architectural approach (e.g., results reported via DB or messaging,
# specific k8s job actions for reset if applicable).

@app.get("/tasks/{task_id}/profits", response_model=TaskProfitResponse)
async def get_task_profits(task_id: str):
    # task_data = active_tasks.get(task_id)
    # if not task_data:
    #     raise HTTPException(status_code=404, detail="Task not found.")
    # trading_instance: Trading = task_data["instance"]
    # try:
    #     profit = trading_instance.profits()
    #     return TaskProfitResponse(task_id=task_id, profit_eur=profit)
    # except Exception as e:
    #     logger.error(f"Error getting profits for task {task_id}: {e}", exc_info=True)
    #     raise HTTPException(status_code=500, detail=f"Error calculating profits: {str(e)}")
    logger.warning(f"Endpoint /tasks/{task_id}/profits is not implemented with TaskManager.")
    raise HTTPException(status_code=501, detail="Not Implemented")


@app.post("/tasks/{task_id}/reset", response_model=TaskResetResponse)
async def reset_task_state(task_id: str):
    # task_data = active_tasks.get(task_id)
    # if not task_data:
    #     raise HTTPException(status_code=404, detail="Task not found.")
    # trading_instance: Trading = task_data["instance"]
    # info_model: TaskInfo = task_data["info_model"]
    # if info_model.status in ["running", "starting", "initializing"] and task_data["thread"].is_alive():
    #      raise HTTPException(status_code=400, detail="Cannot reset a running task. Stop the task first.")
    # try:
    #     trading_instance.reset_trading_state()
    #     info_model.status = "reset" # Update local status
    #     return TaskResetResponse(task_id=task_id, message="Trading task state reset successfully.")
    # except Exception as e:
    #     logger.error(f"Error resetting state for task {task_id}: {e}", exc_info=True)
    #     raise HTTPException(status_code=500, detail=f"Error resetting task state: {str(e)}")
    logger.warning(f"Endpoint /tasks/{task_id}/reset is not implemented with TaskManager.")
    raise HTTPException(status_code=501, detail="Not Implemented")

@app.post("/debug/notify", status_code=200, include_in_schema=True) # include_in_schema for testing
async def debug_send_notification(message: str, channel_id: Optional[str] = None):
    """
    Debug endpoint to test Slack notifications.
    Requires SlackNotifier to be initialized.
    """
    if not slack_notifier_instance:
        raise HTTPException(status_code=503, detail="SlackNotifier not initialized.")

    # Call the synchronous version
    success = send_slack_notification(message, channel_id)
    if success:
        return {"status": "success", "message": "Notification sent."}
    else:
        raise HTTPException(status_code=500, detail="Failed to send notification.")

if __name__ == "__main__":
    # This is for local debugging. For deployment, use Uvicorn directly.
    # e.g., uvicorn trading_service.main:app --reload
    import uvicorn
    logger.info("Attempting to run Uvicorn for trading_service.main:app")
    # Ensure PYTHONPATH is set up if running this directly and imports are module-based
    # For example, if 'config_management' is in the parent directory:
    # import sys
    # sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    uvicorn.run(app, host="0.0.0.0", port=8000)
