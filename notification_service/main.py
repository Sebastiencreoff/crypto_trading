import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from config_management.loader import load_config
from config_management.schemas import AppConfig, SlackConfig as AppSlackConfig # Use the one from AppConfig

from .slack_client import SlackNotifier
from .models import NotificationRequest, NotificationResponse

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Globals / App State ---
app_config: Optional[AppConfig] = None
slack_notifier_instance: Optional[SlackNotifier] = None

# --- FastAPI App Lifespan (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global app_config, slack_notifier_instance
    logger.info("Notification Service starting up...")

    # Load main application configuration
    config_file_path = "config/central_config.json" # Centralized config
    try:
        app_config = load_config(config_file_path)
        logger.info(f"Configuration loaded. Service name: {app_config.service_name}")

        if not app_config.slack:
            logger.error("Slack configuration ('slack' section) not found in the loaded AppConfig.")
            # This is a critical failure for the notification service if Slack is its only medium.
            raise RuntimeError("Slack configuration is missing.")

        # Type cast to ensure it's the SlackConfig model we expect for the notifier
        current_slack_config: AppSlackConfig = app_config.slack

        # Instantiate SlackNotifier
        # The SlackNotifier expects bot_token and default_channel_id.
        # Current AppConfig.slack has webhook_url and channel.
        # SlackNotifier's __init__ has a temporary adaptation:
        # - It interprets slack_config.webhook_url as the bot_token. (Old, now fixed)
        # - It interprets slack_config.channel as the default_channel_id. (Old, now fixed)
        # SlackConfig schema and slack_client.py have been updated.
        logger.info(f"Initializing SlackNotifier with default_channel_id: {current_slack_config.default_channel_id} and bot_token: {str(current_slack_config.bot_token)[:15]}...") # Show only part of token

        slack_notifier_instance = SlackNotifier(slack_config=current_slack_config)

        if not slack_notifier_instance.client:
            logger.error("SlackNotifier client failed to initialize. The service might not function correctly.")
            # Depending on requirements, might raise RuntimeError to stop startup
            # For now, log error and continue; endpoint will fail if notifier is not working.
            # raise RuntimeError("SlackNotifier client failed to initialize.")

        logger.info("SlackNotifier initialized.")
    except Exception as e:
        logger.critical(f"Critical error during Notification Service startup: {e}", exc_info=True)
        # Prevent app from starting if core components fail
        raise RuntimeError(f"Failed to initialize Notification Service: {e}") from e

    yield
    logger.info("Notification Service shutting down...")
    # Cleanup if any (e.g., close connections if SlackNotifier had any persistent ones)
    # WebClient typically doesn't require explicit closing for its HTTP connections.

app = FastAPI(title="Notification Service", version="0.1.0", lifespan=lifespan)

# --- API Endpoints ---

@app.post("/notify", response_model=NotificationResponse)
async def send_notification(request: NotificationRequest):
    """
    Sends a notification message via Slack.
    """
    if not slack_notifier_instance or not slack_notifier_instance.client:
        logger.error("SlackNotifier is not available or not properly initialized.")
        raise HTTPException(status_code=503, detail="Notification service is currently unavailable (Slack client error).")

    logger.info(f"Received notification request for channel '{request.channel_id if request.channel_id else 'default'}': {request.message[:50]}...")

    success = slack_notifier_instance.send_message(
        message=request.message,
        channel_id=request.channel_id # If None, SlackNotifier uses its default
    )

    if success:
        # Use the actual default_channel_id from the notifier's config for logging/response if request.channel_id is None
        target_channel = request.channel_id or slack_notifier_instance.config.default_channel_id
        logger.info(f"Notification sent successfully to channel: {target_channel}.")
        return NotificationResponse(
            status="success",
            details="Message sent to Slack.",
            channel_id=target_channel
        )
    else:
        target_channel = request.channel_id or slack_notifier_instance.config.default_channel_id
        logger.error(f"Failed to send notification to channel: {target_channel}.")
        # Provide a more generic error to the client for security/simplicity
        raise HTTPException(status_code=500, detail="Failed to send notification via Slack.")

if __name__ == "__main__":
    # This is for local debugging. For deployment, use Uvicorn directly.
    # e.g., uvicorn notification_service.main:app --reload
    import uvicorn
    logger.info("Attempting to run Uvicorn for notification_service.main:app")
    # Ensure PYTHONPATH is set up if running this directly and imports are module-based
    uvicorn.run(app, host="0.0.0.0", port=8001) # Running on a different port than trading_service
