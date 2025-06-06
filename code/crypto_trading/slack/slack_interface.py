import logging
import os
import datetime
import tempfile
from slack_sdk import WebClient, RTMClient
from slack_sdk.errors import SlackApiError
from ..plotting import generate_portfolio_graph, generate_pnl_per_trade_graph

logger = logging.getLogger(__name__)

class SlackCommandHandler:
    def __init__(self, conf, task_manager): # conf is expected to be SlackConfig instance
        self.conf = conf # Expected to be an instance of SlackConfig (or similar)
        self.task_manager = task_manager
        self._initialized_successfully = False
        self._keep_running = True # For graceful shutdown

        bot_token = os.environ.get("SLACK_BOT_TOKEN")
        if not bot_token:
            logger.error("SLACK_BOT_TOKEN environment variable not set. SlackCommandHandler cannot initialize.")
            self.client = None
            self.rtm_client = None
            return # Cannot initialize further
        else:
            logger.info("Using bot_token from SLACK_BOT_TOKEN environment variable.")

        try:
            self.client = WebClient(token=bot_token)
            # Test authentication for WebClient
            auth_test_result = self.client.auth_test()
            if not auth_test_result.get("ok"):
                logger.error(f"WebClient authentication failed: {auth_test_result.get('error', 'Unknown error')}")
                self.client = None
                self.rtm_client = None
                return
            logger.info(f"WebClient authenticated successfully for user: {auth_test_result.get('user')}")

            self.rtm_client = RTMClient(token=bot_token)
            # RTMClient doesn't have a direct auth_test equivalent before start()
            # We assume it's okay if WebClient auth worked with the same token.
            self._initialized_successfully = True
            logger.info("SlackCommandHandler RTMClient configured.")

        except SlackApiError as e:
            logger.error(f"Error initializing Slack clients: {e.response['error']}", exc_info=True)
            self.client = None
            self.rtm_client = None
            return
        except Exception as e:
            logger.error(f"An unexpected error occurred during Slack client initialization: {e}", exc_info=True)
            self.client = None
            self.rtm_client = None
            return

        # Define a default task configuration template
        self.default_task_config = {
            "app_settings": {
                "notification_service_url": None,
                "base_config_path": "/app/config",
                "database_url": None,
                "other_settings": {"delay_seconds": 5}
            },
            "exchange_settings": {
                "name": "simulation",
                "api_key": None,
                "secret_key": None,
                "extra_settings": {"dir_path": "/app/config/sample_data/simu_data_btc_1m.csv"},
                "paper_trade": True
            },
            "algo_settings": {
                "name": "moving_average_crossover",
                "parameters": {"short_window": 2, "long_window": 4}
            },
            "task_parameters": {
                "currency": "BTC/USD",
                "transaction_amount": 10,
                "interval": "1m"
            }
        }
        # Allow overriding TASK_DOCKER_IMAGE via conf (passed as self.conf.task_docker_image perhaps)
        # For now, this is just illustrative as TaskManager uses its own constant.
        if hasattr(self.conf, 'task_docker_image') and self.conf.task_docker_image:
            self.task_docker_image = self.conf.task_docker_image
        else:
            # Fallback if not in conf, though run_slack_handler doesn't set this on SlackConfig
            self.task_docker_image = os.environ.get("TASK_DOCKER_IMAGE", "your-docker-registry/trading-task:latest")

        if self._initialized_successfully:
            logger.info(f"SlackCommandHandler initialized. Task Docker Image (for reference): {self.task_docker_image}")
        else:
            logger.error("SlackCommandHandler initialization failed.")

    def is_initialized(self):
        """Checks if the Slack clients were initialized successfully."""
        return self._initialized_successfully

    def stop(self):
        """Signals the listener to stop and attempts to close RTM client."""
        logger.info("Stopping SlackCommandHandler...")
        self._keep_running = False
        if self.rtm_client and hasattr(self.rtm_client, '_stop'): # _stop is internal, use with caution
            # This is not standard API, RTMClient.start() is blocking.
            # A more robust way would be to run RTM in a thread that can be joined.
            logger.info("Attempting to signal RTM client to stop (experimental).")
            # self.rtm_client._stop() # This is hypothetical, not a public API
        # For now, the main loop in start_listening needs to check self._keep_running

    def start_listening(self):
        if not self.is_initialized():
            logger.error("SlackCommandHandler is not initialized. Cannot start listening.")
            return

        logger.info("Starting to listen for Slack events...")

        # The RTMClient.start() is blocking. To make it stoppable by self._keep_running,
        # we would need to implement a custom run loop or run RTMClient in a thread.
        # For now, we'll add the @self.rtm_client.on decorator and then call start().
        # The self._keep_running flag is more for a future refactor of this part.

        @self.rtm_client.on("message")
        def handle_message(client: RTMClient, event: dict):
            if "text" in event and "user" in event and "channel" in event:
                command_text = event["text"]
                user_id = event["user"]
                channel_id = event["channel"]

                # Basic command parsing (e.g., commands start with a specific prefix)
                # This is a placeholder and will need to be more robust.
                if command_text.startswith("!crypto"): # Example prefix
                    actual_command = command_text.split("!crypto", 1)[1].strip()
                    if actual_command:
                        logger.info(f"Received command: '{actual_command}' from user '{user_id}' in channel '{channel_id}'")
                        self.handle_command(actual_command, user_id, channel_id)
                    else:
                        logger.info(f"Received empty command from user '{user_id}' in channel '{channel_id}'")
                # else:
                #     logger.debug(f"Ignoring non-command message from user '{user_id}' in channel '{channel_id}'")

        try:
            # Note: RTMClient.start() is a blocking call.
            # The self._keep_running flag won't stop this loop directly without modifying how RTMClient is used.
            # (e.g. by running it in a thread and having the thread check the flag, or using an async version)
            if self._keep_running: # Check flag before starting (mostly symbolic for current RTMClient.start() blocking nature)
                self.rtm_client.start()
            else:
                logger.info("Skipping RTM client start as stop signal was received.")
        except SlackApiError as e:
            logger.error(f"Error starting RTM client: {e.response['error']}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while starting RTM client: {e}")
        finally:
            logger.info("RTM client event loop finished.")


    def handle_command(self, command, user, channel):
        """
        Handles a parsed command.
        """
        logger.info(f"Handling command: '{command}' from user '{user}' in channel '{channel}'")

        parts = command.lower().split()
        main_command = parts[0] if parts else ""
        args = parts[1:]
        response_message = ""

        if not self.task_manager:
            logger.error("TaskManager not available to SlackCommandHandler.")
            response_message = "Error: TaskManager is not configured. Cannot process commands."
            if channel and response_message:
                 self._send_message(channel, response_message)
            return

        if main_command == "help":
            response_message = self._handle_help_command()
        elif main_command == "start":
            response_message = self._handle_start_command(args, user, channel)
        elif main_command == "stop":
            response_message = self._handle_stop_command(args, user, channel)
        elif main_command == "status":
            response_message = self._handle_status_command(args, user, channel)
        elif main_command == "graph":
            # This command handles its own Slack messaging due to file upload
            self._handle_graph_command(args, user, channel)
            response_message = None # Avoid double messaging
        elif main_command == "pnl_chart":
            # This command handles its own Slack messaging due to file upload
            self._handle_pnl_chart_command(args, user, channel)
            response_message = None # Avoid double messaging
        else:
            response_message = f"Unknown command: '{main_command}'. Type `!crypto help` for available commands."
            logger.warning(f"Unknown command: '{main_command}' from user '{user}' in channel '{channel}'")

        if response_message and channel: # Some handlers might send messages themselves
            self._send_message(channel, response_message)

    def _send_message(self, channel, text):
        try:
            self.client.chat_postMessage(channel=channel, text=text)
        except SlackApiError as e:
            logger.error(f"Error posting message to Slack channel {channel}: {e.response['error']}")

    def _handle_help_command(self):
        return (
            "Available commands:\n"
            "- `!crypto help`: Shows this help message.\n"
            "- `!crypto start`: Starts a new trading task with default configuration.\n"
            # TODO: Add way to specify config for start, e.g., !crypto start <config_name_or_json>
            "- `!crypto stop <task_id>`: Stops the trading task with the given ID.\n"
            "- `!crypto status`: Lists all active tasks and their statuses.\n"
            "- `!crypto status <task_id>`: Shows the status of the specific trading task.\n"
            "- `!crypto graph <task_id>`: Generates and uploads a portfolio value graph for the task.\n"
            "- `!crypto pnl_chart <task_id>`: Generates and uploads a P/L per trade chart for the task."
        )

    def _fetch_portfolio_history(self, task_id, task_results):
        """
        Placeholder/Simulated data fetching for portfolio history.
        Ideally, this would parse task_results or query a database.
        """
        logger.info(f"Fetching portfolio history for task_id: {task_id}. Using simulated data.")
        # task_results are logs for now, so not directly usable for structured data yet.
        # Simulate data: a list of (datetime, value) tuples
        return [(datetime.datetime.now() - datetime.timedelta(days=x), 1000 + x * 10 - x**2) for x in range(10, 0, -1)]

    def _fetch_trade_history(self, task_id, task_results):
        """
        Placeholder/Simulated data fetching for trade history.
        Ideally, this would parse task_results (logs) or query a database.
        """
        logger.info(f"Fetching trade history for task_id: {task_id}. Using simulated data.")
        # task_results are logs for now.
        # Simulate data: list of dicts with 'label' and 'profit'
        return [{'label': f'Trade{x+1}', 'profit': (x % 3 - 1) * (x+1) * 10} for x in range(8)]


    def _handle_graph_command(self, args, user, channel):
        logger.info(f"Handling 'graph' command from user '{user}' in channel '{channel}' with args: {args}")
        if not args:
            self._send_message(channel, "Error: `task_id` is required. Usage: `!crypto graph <task_id>`")
            return

        task_id = args[0]

        # 1. Fetch task results (currently logs, but future might hold structured data path)
        task_results = self.task_manager.get_task_results(task_id)
        if task_results is None: # Assuming get_task_results returns None if task not found or no results
            self._send_message(channel, f"Error: Could not retrieve results for task `{task_id}`. It may not exist or has no output yet.")
            return

        # 2. Fetch (simulated) portfolio data
        # In a real scenario, task_results might be logs to parse or a path to a data file
        portfolio_data = self._fetch_portfolio_history(task_id, task_results)
        if not portfolio_data:
            self._send_message(channel, f"No portfolio data found or generated for task `{task_id}`.")
            return

        # 3. Generate graph
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                temp_file_path = tmpfile.name

            graph_path = generate_portfolio_graph(portfolio_data, temp_file_path)

            if not graph_path:
                self._send_message(channel, f"Error: Failed to generate portfolio graph for task `{task_id}`.")
                return

            # 4. Upload graph
            logger.info(f"Uploading graph from {graph_path} for task {task_id} to channel {channel}")
            upload_response = self.client.files_upload_v2(
                channel=channel,
                filepath=graph_path,
                title=f"Portfolio Value Over Time - Task {task_id}",
                initial_comment=f"Portfolio graph for task `{task_id}`:"
            )
            if upload_response.get("ok"):
                logger.info(f"Graph for task {task_id} uploaded successfully.")
                # No need to send another message, the file upload itself is the message.
            else:
                logger.error(f"Failed to upload graph for task {task_id}: {upload_response.get('error', 'Unknown error')}")
                self._send_message(channel, f"Error: Failed to upload portfolio graph for task `{task_id}`.")

        except Exception as e:
            logger.error(f"An error occurred during graph command for task {task_id}: {e}", exc_info=True)
            self._send_message(channel, f"An unexpected error occurred while generating the graph for task `{task_id}`.")
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logger.debug(f"Temporary graph file {temp_file_path} deleted.")

    def _handle_pnl_chart_command(self, args, user, channel):
        logger.info(f"Handling 'pnl_chart' command from user '{user}' in channel '{channel}' with args: {args}")
        if not args:
            self._send_message(channel, "Error: `task_id` is required. Usage: `!crypto pnl_chart <task_id>`")
            return

        task_id = args[0]

        task_results = self.task_manager.get_task_results(task_id)
        if task_results is None:
            self._send_message(channel, f"Error: Could not retrieve results for task `{task_id}`. It may not exist or has no output yet.")
            return

        trade_data = self._fetch_trade_history(task_id, task_results)
        if not trade_data:
            self._send_message(channel, f"No trade data found or generated for task `{task_id}`.")
            return

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                temp_file_path = tmpfile.name

            chart_path = generate_pnl_per_trade_graph(trade_data, temp_file_path)

            if not chart_path:
                self._send_message(channel, f"Error: Failed to generate P/L chart for task `{task_id}`.")
                return

            logger.info(f"Uploading P/L chart from {chart_path} for task {task_id} to channel {channel}")
            upload_response = self.client.files_upload_v2(
                channel=channel,
                filepath=chart_path,
                title=f"P/L Per Trade - Task {task_id}",
                initial_comment=f"P/L per trade chart for task `{task_id}`:"
            )
            if upload_response.get("ok"):
                logger.info(f"P/L chart for task {task_id} uploaded successfully.")
            else:
                logger.error(f"Failed to upload P/L chart for task {task_id}: {upload_response.get('error', 'Unknown error')}")
                self._send_message(channel, f"Error: Failed to upload P/L chart for task `{task_id}`.")

        except Exception as e:
            logger.error(f"An error occurred during pnl_chart command for task {task_id}: {e}", exc_info=True)
            self._send_message(channel, f"An unexpected error occurred while generating the P/L chart for task `{task_id}`.")
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logger.debug(f"Temporary P/L chart file {temp_file_path} deleted.")

    def _handle_start_command(self, args, user, channel):
        logger.info(f"Handling 'start' command from user '{user}' in channel '{channel}' with args: {args}")
        # For now, use the default_task_config.
        # Future: allow specifying a config name or passing JSON for config_obj.
        config_obj = self.default_task_config.copy()

        # Potentially, allow overriding parts of config_obj via args if we define a syntax.
        # For example: !crypto start currency=ETH/USD interval=5m
        # This is left for future enhancement.

        logger.info(f"Using task configuration: {config_obj}")

        task_id = self.task_manager.create_task(config_obj)
        if task_id:
            logger.info(f"Task created successfully with ID: {task_id} by user '{user}'")
            return f"Trading task started successfully. Task ID: `{task_id}`"
        else:
            logger.error(f"Failed to create task for user '{user}'")
            return "Error: Failed to start trading task. Check logs for details."

    def _handle_stop_command(self, args, user, channel):
        logger.info(f"Handling 'stop' command from user '{user}' in channel '{channel}' with args: {args}")
        if not args:
            return "Error: `task_id` is required for the stop command. Usage: `!crypto stop <task_id>`"

        task_id = args[0]
        logger.info(f"Attempting to stop task with ID: {task_id} as requested by user '{user}'")

        success = self.task_manager.stop_task(task_id)
        if success:
            logger.info(f"Stop signal sent for task ID: {task_id} by user '{user}'")
            return f"Stop signal sent for task `{task_id}`. It may take a moment to terminate."
        else:
            logger.error(f"Failed to send stop signal for task ID: {task_id} (user: '{user}')")
            return f"Error: Failed to stop task `{task_id}`. It might have already completed or does not exist. Check logs."

    def _handle_status_command(self, args, user, channel):
        logger.info(f"Handling 'status' command from user '{user}' in channel '{channel}' with args: {args}")
        if args: # Specific task ID provided
            task_id = args[0]
            logger.info(f"Fetching status for specific task ID: {task_id} for user '{user}'")
            status = self.task_manager.get_task_status(task_id)
            if status:
                return f"Status for task `{task_id}`: {status}"
            else:
                return f"Error: Could not retrieve status for task `{task_id}`. It may not exist or an error occurred."
        else: # List all tasks
            logger.info(f"Fetching status for all tasks for user '{user}'")
            tasks = self.task_manager.list_tasks()
            if not tasks:
                return "No active tasks found."

            status_messages = [f"- Task `{task_id}`: {status}" for task_id, status in tasks.items()]
            return "Current tasks and their statuses:\n" + "\n".join(status_messages)

if __name__ == '__main__':
    # This is example usage and would require a conf object and a task_manager mock/instance
    # For now, it just demonstrates that the logger works.
    logging.basicConfig(level=logging.INFO)
    logger.info("SlackCommandHandler module directly executed (for testing/example).")

    # Example: To run this, you would need to set SLACK_BOT_TOKEN environment variable
    # and provide a mock or real conf and task_manager.
    # For example:
    # os.environ["SLACK_BOT_TOKEN"] = "xoxb-your-token"
    # mock_conf = {} # Replace with a proper conf object or mock
    # mock_task_manager = None # Replace with a proper task_manager object or mock
    # handler = SlackCommandHandler(conf=mock_conf, task_manager=mock_task_manager)
    # handler.start_listening() # This would block, so typically run in a thread or async context
    # logger.info("Exiting example execution.")
