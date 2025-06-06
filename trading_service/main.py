import logging
import uuid
import threading
import queue
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from typing import Dict, Optional

# Adjust imports based on project structure
# Assuming config_management, trading_service, crypto_trading are top-level or in PYTHONPATH
from config_management.loader import load_config
from config_management.schemas import AppConfig, ExchangeConfig, AlgoConfig
from crypto_trading.database.core_operations import get_session as get_db_session_from_engine # Renamed to avoid conflict
from crypto_trading.config import init as init_global_config, app_config as global_app_config # For DB engine init

from .core import Trading
from .models import CreateTaskRequest, TaskStatusResponse, TaskInfo, TaskCreateResponse, TaskStopResponse, TaskProfitResponse, TaskResetResponse

# --- Globals / App State ---
# TODO: Replace with a more robust task management and persistence layer if needed
active_tasks: Dict[str, Dict[str, any]] = {}  # Stores Trading instances, threads, stop_events, queues
app_config: Optional[AppConfig] = None # Loaded config

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI App Lifespan (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global app_config
    logger.info("FastAPI application starting up...")
    # Load configuration
    # The path to config might be from an env variable or a fixed path
    # For now, using the one specified in the overall task
    config_file_path = "config/central_config.json"
    try:
        # Initialize the global config from crypto_trading.config first to set up the DB engine
        # This is a bit of a workaround as crypto_trading.config.init also loads AppConfig.
        # Ideally, DB engine setup should be decoupled or handled more centrally.
        init_global_config(config_file_path) # This sets up global_app_config and its DB engine
        app_config = global_app_config # Use the already loaded config
        logger.info(f"Configuration loaded successfully for service: {app_config.service_name}")

        if not app_config:
             logger.error("Failed to load application configuration during startup.")
             # This should ideally prevent startup or signal critical failure
             raise RuntimeError("Failed to load application configuration.")

    except Exception as e:
        logger.error(f"Critical error during application startup: {e}", exc_info=True)
        # Depending on policy, might re-raise to stop app or try to run in a degraded state
        raise
    yield
    logger.info("FastAPI application shutting down...")
    # Cleanup: signal all active tasks to stop
    logger.info("Application shutdown: Processing active tasks...")
    for task_id, task_data in list(active_tasks.items()): # Iterate over a copy for safe removal
        thread_instance = task_data.get("thread")
        stop_event_instance = task_data.get("stop_event")

        if isinstance(stop_event_instance, threading.Event) and not stop_event_instance.is_set():
            logger.info(f"Shutdown: Signalling task {task_id} to stop.")
            stop_event_instance.set()

        if isinstance(thread_instance, threading.Thread) and thread_instance.is_alive():
            logger.info(f"Shutdown: Waiting for task {task_id} thread to join.")
            thread_instance.join(timeout=10) # Wait for 10 seconds
            if thread_instance.is_alive():
                logger.warning(f"Shutdown: Task {task_id} thread did not join in time.")
            else:
                logger.info(f"Shutdown: Task {task_id} thread joined. Session should be closed by wrapper.")
        else:
            logger.info(f"Shutdown: Task {task_id} thread was not alive or not a thread instance.")
            # If thread is not alive, its session should ideally be closed by the wrapper.
            # However, if it died unexpectedly, the session might be orphaned.
            # Closing it here is a fallback, but be cautious if thread could still be running.
            # Since threads are daemonic, they might be abruptly stopped on exit without finally running.
            session_to_close = task_data.get("db_session")
            if session_to_close:
                logger.warning(f"Shutdown: Task {task_id} thread not alive, attempting fallback session close.")
                try:
                    session_to_close.close()
                    logger.info(f"Shutdown: Fallback session closed for task {task_id}.")
                    task_data["db_session"] = None # Mark as closed
                except Exception as e_close_fallback:
                    logger.error(f"Shutdown: Error during fallback session close for task {task_id}: {e_close_fallback}", exc_info=True)

        # Optionally, remove from active_tasks if confirmed done.
        # if not thread_instance.is_alive():
        #     active_tasks.pop(task_id, None)

    logger.info("Shutdown complete.")


app = FastAPI(title="Trading Service", version="0.1.0", lifespan=lifespan)

# --- Dependency for SQLAlchemy Session ---
# This creates a new session per request. For background tasks, session management needs care.
def get_db_session():
    if not global_app_config or not global_app_config.database:
        logger.error("Database configuration not available in global_app_config.")
        raise HTTPException(status_code=500, detail="Database not configured")

    # get_db_session_from_engine() is expected to create a new session from the engine
    # initialized by crypto_trading.config.init()
    session = get_db_session_from_engine()
    try:
        yield session
    finally:
        session.close()

# --- API Endpoints ---

@app.post("/tasks", response_model=TaskCreateResponse, status_code=201)
async def create_trading_task(task_request: CreateTaskRequest, session = Depends(get_db_session)):
    """
    Create and launch a new trading task.
    """
    if not app_config:
        raise HTTPException(status_code=500, detail="Server configuration not loaded.")

    task_id = str(uuid.uuid4())
    logger.info(f"Received request to create task {task_id} for {task_request.currency_pair} on {task_request.exchange_name} with algo {task_request.algo_name}")

    # Find the selected exchange configuration
    selected_exchange: Optional[ExchangeConfig] = None
    for ex_cfg in app_config.exchanges:
        if ex_cfg.name == task_request.exchange_name:
            selected_exchange = ex_cfg
            break
    if not selected_exchange:
        logger.error(f"Task {task_id}: Exchange '{task_request.exchange_name}' not found in configuration.")
        raise HTTPException(status_code=400, detail=f"Exchange '{task_request.exchange_name}' not configured.")

    # Find the selected algorithm configuration
    selected_algo_config: Optional[AlgoConfig] = None
    if app_config.algorithms:
        for algo_cfg in app_config.algorithms:
            if algo_cfg.name == task_request.algo_name:
                selected_algo_config = algo_cfg
                break
    if not selected_algo_config:
        logger.error(f"Task {task_id}: Algorithm '{task_request.algo_name}' not found in configuration.")
        raise HTTPException(status_code=400, detail=f"Algorithm '{task_request.algo_name}' not configured.")

    # Handle algorithm parameter overrides
    if task_request.algo_override_params:
        # Create a copy to not modify the global config
        temp_params = selected_algo_config.parameters.copy()
        temp_params.update(task_request.algo_override_params)
        # Create a temporary AlgoConfig with overridden params
        # Note: This doesn't change the global app_config, only for this task
        effective_algo_config = AlgoConfig(name=selected_algo_config.name, parameters=temp_params)
    else:
        effective_algo_config = selected_algo_config

    task_params = {
        "currency": task_request.currency_pair,
        "transaction_amount": task_request.transaction_amount
    }

    stop_event = threading.Event()
    results_queue = queue.Queue() # For communication from Trading thread if needed

    # The SQLAlchemy session for the Trading class instance:
    # It's tricky because the session from Depends(get_db_session) is request-scoped.
    # A background thread needs its own session or a session passed to it that remains valid.
    # For simplicity here, we create a new session for the thread.
    # This means the thread is responsible for closing it.
    # A more robust solution might involve a session factory or careful session management.
    thread_session = get_db_session_from_engine()


    try:
        trading_instance = Trading(
            app_config=app_config,
            exchange_config=selected_exchange,
            algo_config=effective_algo_config,
            task_params=task_params,
            session=thread_session, # Pass the new session to the trading instance
            task_id=task_id,
            stop_event=stop_event,
            results_queue=results_queue
        )
    except Exception as e:
        thread_session.close() # Clean up session if Trading init fails
        logger.error(f"Task {task_id}: Failed to initialize Trading instance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize trading task: {e}")

    # Wrapper function to ensure session closure for the trading task thread
    def task_runner_wrapper(instance: Trading, session_to_close: any, task_id_for_log: str):
        try:
            logger.info(f"Task {task_id_for_log}: Thread starting.")
            instance.run()
            logger.info(f"Task {task_id_for_log}: Run method completed.")
        except Exception as e_run:
            logger.error(f"Task {task_id_for_log}: Exception in run method: {e_run}", exc_info=True)
            # Optionally update task status here if results_queue is accessible or via another mechanism
        finally:
            logger.info(f"Task {task_id_for_log}: Thread finished. Closing database session.")
            if session_to_close:
                try:
                    session_to_close.close()
                    logger.info(f"Task {task_id_for_log}: Database session closed by wrapper.")
                except Exception as e_close:
                    logger.error(f"Task {task_id_for_log}: Error closing database session in wrapper: {e_close}", exc_info=True)

    # Run Trading.run in a background thread using the wrapper
    try:
        thread = threading.Thread(target=task_runner_wrapper, args=(trading_instance, thread_session, task_id), daemon=True)
        thread.start()
    except Exception as e:
        # If thread failed to start, close the session immediately
        logger.error(f"Task {task_id}: Failed to start Trading thread: {e}", exc_info=True)
        if thread_session:
            try:
                thread_session.close()
            except Exception as e_close_immediate:
                logger.error(f"Task {task_id}: Error closing session immediately after thread start failure: {e_close_immediate}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start trading thread: {e}")


    task_info_model = TaskInfo(
        task_id=task_id,
        currency_pair=task_request.currency_pair,
        exchange_name=task_request.exchange_name,
        algo_name=task_request.algo_name,
        status="starting"
    )

    active_tasks[task_id] = {
        "info_model": task_info_model, # Store the Pydantic model for status
        "instance": trading_instance, # Keep instance for methods like stop(), profits()
        "thread": thread,
        "stop_event": stop_event,
        "results_queue": results_queue, # Store queue to potentially retrieve messages
        "db_session": thread_session # Store session to close it when task is fully stopped/cleaned up
    }
    logger.info(f"Task {task_id} for {task_request.currency_pair} created and thread started.")

    status_response = TaskStatusResponse(
        task_id=task_id,
        status="starting", # Initial status
        currency_pair=task_request.currency_pair,
        exchange_name=task_request.exchange_name,
        algo_name=task_request.algo_name,
        message="Trading task successfully started."
    )
    return TaskCreateResponse(task_id=task_id, message="Trading task created.", details=status_response)


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get the status of a specific trading task.
    """
    task_data = active_tasks.get(task_id)
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found.")

    info_model: TaskInfo = task_data["info_model"]

    # Check the results queue for the latest status message if any
    # This is a simple way to get some feedback from the thread.
    # More sophisticated status would involve the Trading class updating its status in a shared way.
    current_status_message = "Status based on last known operation."
    try:
        # Non-blocking get from queue
        while not task_data["results_queue"].empty():
            msg = task_data["results_queue"].get_nowait()
            if isinstance(msg, dict):
                info_model.status = msg.get("status", info_model.status)
                current_status_message = msg.get("message", current_status_message)
                # Update status in our active_tasks dict
                task_data["info_model"].status = info_model.status

    except queue.Empty:
        pass # No new messages

    # Check if thread is alive
    thread: threading.Thread = task_data["thread"]
    if not thread.is_alive() and info_model.status not in ["stopped", "error", "simulation_ended", "endofprocess"]:
        # If thread died unexpectedly, mark as error or unknown
        logger.warning(f"Task {task_id} thread is not alive but status is {info_model.status}. Marking as 'unknown_stopped'.")
        info_model.status = "unknown_stopped"
        current_status_message = "Trading thread is no longer running."


    return TaskStatusResponse(
        task_id=task_id,
        status=info_model.status,
        currency_pair=info_model.currency_pair,
        exchange_name=info_model.exchange_name,
        algo_name=info_model.algo_name,
        message=current_status_message
    )

@app.post("/tasks/{task_id}/stop", response_model=TaskStopResponse)
async def stop_trading_task(task_id: str):
    task_data = active_tasks.get(task_id)
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found.")

    trading_instance: Trading = task_data["instance"]
    stop_event: threading.Event = task_data["stop_event"]
    thread: threading.Thread = task_data["thread"]
    info_model: TaskInfo = task_data["info_model"]

    if stop_event.is_set() and not thread.is_alive():
        message = "Task has already been stopped and thread terminated."
        info_model.status = "stopped" # Ensure status reflects this
    elif stop_event.is_set():
        message = "Task is already in the process of stopping."
        info_model.status = "stopping"
    else:
        logger.info(f"Sending stop signal to task {task_id}")
        trading_instance.stop() # This sets the event
        message = "Stop signal sent. Task will process shutdown."
        info_model.status = "stopping"

    # If the thread is already dead (e.g. crashed), try to clean up its session.
    # The task_runner_wrapper should handle this for normal exits.
    # This is a fallback for unexpected thread termination.
    if not thread.is_alive() and task_data.get("db_session"):
        logger.warning(f"Task {task_id} thread is not alive. Attempting to close its DB session post-stop signal.")
        try:
            task_data["db_session"].close()
            logger.info(f"Task {task_id}: DB session closed post-stop signal for non-alive thread.")
            task_data["db_session"] = None # Mark as closed
        except Exception as e_close_stop:
            logger.error(f"Task {task_id}: Error closing DB session post-stop for non-alive thread: {e_close_stop}", exc_info=True)

    return TaskStopResponse(task_id=task_id, message=message)


# TODO: Add endpoints for /tasks/{task_id}/profits and /tasks/{task_id}/reset
# These would call trading_instance.profits() and trading_instance.reset_trading_state()
# Ensure session management is correct for these calls if they interact with DB.
# The trading_instance already has its own session.

@app.get("/tasks/{task_id}/profits", response_model=TaskProfitResponse)
async def get_task_profits(task_id: str):
    task_data = active_tasks.get(task_id)
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found.")

    trading_instance: Trading = task_data["instance"]
    # The trading_instance uses its own dedicated session.
    try:
        profit = trading_instance.profits()
        return TaskProfitResponse(task_id=task_id, profit_eur=profit)
    except Exception as e:
        logger.error(f"Error getting profits for task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error calculating profits: {str(e)}")


@app.post("/tasks/{task_id}/reset", response_model=TaskResetResponse)
async def reset_task_state(task_id: str):
    task_data = active_tasks.get(task_id)
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found.")

    trading_instance: Trading = task_data["instance"]
    info_model: TaskInfo = task_data["info_model"]

    if info_model.status in ["running", "starting", "initializing"] and task_data["thread"].is_alive():
         raise HTTPException(status_code=400, detail="Cannot reset a running task. Stop the task first.")

    try:
        # The trading_instance uses its own dedicated session.
        trading_instance.reset_trading_state()
        info_model.status = "reset" # Update local status
        return TaskResetResponse(task_id=task_id, message="Trading task state reset successfully.")
    except Exception as e:
        logger.error(f"Error resetting state for task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error resetting task state: {str(e)}")


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
