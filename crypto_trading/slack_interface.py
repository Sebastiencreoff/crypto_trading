import logging
import time
import os
import tempfile
from slack_sdk import WebClient
from slack_sdk.rtm import RTMClient # Corrected import for RTMClient
from slack_sdk.errors import SlackApiError

# Make sure to install slack_sdk with RTMClient capabilities if not already:
# pip install "slack_sdk[rtm]" or ensure it's in setup.py

from crypto_trading.plotting import generate_portfolio_graph, generate_pnl_per_trade_graph
from crypto_trading import model


class SlackInterface:
    def __init__(self, conf, trading_instance):
        """
        Initializes the SlackInterface.

        Args:
            conf: The application configuration object.
            trading_instance: An instance of the Trading class.
        """
        self.conf = conf
        self.trading_instance = trading_instance
        self.slack_token = getattr(self.conf, 'slack_token', None)
        self.channel_id = getattr(self.conf, 'slack_channel_id', None)

        if not self.slack_token or not self.channel_id:
            logging.error("Slack token or channel ID is not configured. SlackInterface will be disabled.")
            self.client = None
            self.rtm_client = None
            return

        self.client = WebClient(token=self.slack_token)
        self.rtm_client = RTMClient(token=self.slack_token)
        # Test connection during initialization to fail early if token is invalid
        try:
            auth_test = self.client.auth_test()
            if not auth_test["ok"]:
                logging.error(f"Slack WebClient auth_test failed: {auth_test['error']}. Disabling SlackInterface.")
                self.client = None
                self.rtm_client = None
                return
            self.bot_id = auth_test.get("bot_id") or auth_test.get("user_id") # bot_id for newer apps, user_id for classic
            logging.info(f"SlackInterface initialized successfully. Bot ID: {self.bot_id}")
        except Exception as e:
            logging.error(f"Error during Slack WebClient auth_test: {e}. Disabling SlackInterface.")
            self.client = None
            self.rtm_client = None


    def send_message(self, text: str):
        """
        Sends a message to the configured Slack channel.

        Args:
            text: The message text to send.
        """
        if not self.client or not self.channel_id:
            logging.warning("Slack client not initialized or channel_id missing. Cannot send message.")
            return
        try:
            response = self.client.chat_postMessage(channel=self.channel_id, text=text)
            if not response["ok"]:
                logging.error(f"Error sending Slack message: {response['error']}")
        except Exception as e:
            logging.error(f"Exception while sending Slack message: {e}")

    def handle_command(self, command_text: str, user_id: str, channel: str):
        """
        Handles a command received from Slack.

        Args:
            command_text: The text of the command received.
            user_id: The ID of the user who sent the command.
            channel: The ID of the channel where the command was received.
        """
        if not self.client:
            logging.warning("Slack client not initialized. Cannot handle command.")
            return

        command = command_text.lower().strip()
        response_text = ""

        if command.startswith("start"):
            try:
                # Assuming trading_instance.run() is non-blocking or handled elsewhere.
                # If trading_instance.run() is blocking and not in its own thread,
                # this will block the Slack listener.
                # main.py should ensure trading.run() is in the main thread
                # and start_listening is in a separate thread.
                if not self.trading_instance.is_running():
                    # Potentially, re-initialize or start the trading instance's loop
                    # This part needs careful consideration of how trading.run() is structured.
                    # For now, we assume it can be "re-triggered" if stopped, or it's already running.
                    # A direct call to self.trading_instance.run() might be problematic if it's not designed for re-entry
                    # or if it's already running in the main thread.
                    # A more robust approach for a 'start' command might be to set a flag
                    # that the main trading loop checks, or to re-initialize the trading components.
                    # Let's assume for now it signals the main loop to start if it was stopped.
                    # This is a conceptual 'start'. The current trading.run() starts a loop.
                    # If stop() just sets self.loop = 0, start should set self.loop = 1
                    # and potentially re-initialize if necessary.
                    # The existing `trading.run()` is a blocking loop.
                    # A 'start' command is tricky if the main loop in `trading.run()` has exited.
                    # We'll assume 'start' is to ensure it *is* running, if already running, it's a NOP.
                    # If it was stopped via `stop()`, `trading.run()` would have exited.
                    # This command is more like a "ensure running" or "restart".
                    # For simplicity, if it's stopped, we can't just call run() again from here easily
                    # without major refactoring of Trading class.
                    # So, we'll just report status.
                    if not self.trading_instance.is_running():
                         response_text = "Trading bot is currently stopped. A 'start' command from Slack cannot restart a fully exited trading loop without a redesign. Please restart the application if needed."
                    else:
                        # This case implies trading_instance.run() was designed to be callable multiple times
                        # or that stop() just paused it. Given the current trading.py, stop() makes run() exit.
                        # self.trading_instance.run() # This is problematic.
                        response_text = "Trading bot is already running or cannot be restarted from here."
                else:
                    response_text = "Trading bot is already running."
                self.send_message(response_text)
            except Exception as e:
                logging.error(f"Error processing 'start' command: {e}")
                self.send_message(f"Error processing 'start' command: {e}")
        elif command.startswith("stop"):
            try:
                self.trading_instance.stop()
                response_text = "Trading bot stop signal sent."
                self.send_message(response_text)
            except Exception as e:
                logging.error(f"Error processing 'stop' command: {e}")
                self.send_message(f"Error processing 'stop' command: {e}")
        elif command.startswith("status"):
            try:
                status = self.trading_instance.is_running()
                profits = self.trading_instance.profits() # Assuming this method exists and is thread-safe
                response_text = f"Bot status: {'Running' if status else 'Stopped'}. Current profits: {profits}"
                self.send_message(response_text)
            except Exception as e:
                logging.error(f"Error processing 'status' command: {e}")
                self.send_message(f"Error processing 'status' command: {e}")

        elif command.startswith("graph"):
            logging.info(f"Graph command received from user {user_id} in channel {channel}.")
            temp_file_path = None
            try:
                initial_capital = self.conf.initial_capital
                data_points = model.get_portfolio_value_history(initial_capital)

                if not data_points or len(data_points) < 2:
                    self.send_message("Not enough data to generate a graph yet.")
                    return

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    temp_file_path = tmp.name

                logging.debug(f"Temporary file for graph: {temp_file_path}")
                graph_file_path_generated = generate_portfolio_graph(data_points, temp_file_path)

                if graph_file_path_generated:
                    self.client.files_upload_v2(
                        channel=channel,
                        file=graph_file_path_generated,
                        title="Portfolio Value Over Time",
                        initial_comment="Here is the portfolio value graph:"
                    )
                    logging.info(f"Successfully uploaded graph to Slack channel {channel}")
                else:
                    self.send_message("Sorry, I couldn't generate the portfolio graph at this time (generation failed).")

            except SlackApiError as e:
                logging.error(f"Slack API error during graph command: {e.response['error'] if e.response else str(e)}")
                self.send_message(f"Sorry, a Slack error occurred while uploading the graph: {e.response['error'] if e.response else str(e)}")
            except Exception as e:
                logging.error(f"Unexpected error during graph command: {e}", exc_info=True)
                self.send_message("Sorry, an unexpected error occurred while processing the graph command.")
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                        logging.debug(f"Successfully removed temporary graph file: {temp_file_path}")
                    except OSError as e_remove: # Renamed to avoid conflict with outer 'e'
                        logging.error(f"Error removing temporary graph file {temp_file_path}: {e_remove}")

        elif command.startswith("pnl_chart"):
            logging.info(f"P/L Chart command received from user {user_id} in channel {channel}.")
            temp_file_path = None
            try:
                completed_trades = list(model.Trading.select(
                    model.Trading.q.sell_date_time != None,
                    orderBy=model.Trading.q.sell_date_time
                ))

                if not completed_trades:
                    self.send_message("No completed trades found to generate a P/L chart.")
                    return

                trades_data_for_plot = []
                for i, trade in enumerate(completed_trades):
                    trade_label = trade.sell_date_time.strftime('%Y-%m-%d %H:%M') if trade.sell_date_time else f"Trade {i+1}"
                    trades_data_for_plot.append({
                        'label': trade_label,
                        'profit': trade.profit if trade.profit is not None else 0.0
                    })

                if not trades_data_for_plot: # Should not happen if completed_trades is not empty, but as a safeguard
                    self.send_message("Could not process trade data for P/L chart.")
                    return

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    temp_file_path = tmp.name

                logging.debug(f"Temporary file for P/L chart: {temp_file_path}")
                graph_file_path = generate_pnl_per_trade_graph(trades_data_for_plot, temp_file_path)

                if graph_file_path:
                    self.client.files_upload_v2(
                        channel=channel,
                        file=graph_file_path,
                        title="Profit/Loss per Trade",
                        initial_comment="Here is the P/L chart for completed trades:"
                    )
                    logging.info(f"Successfully uploaded P/L chart to Slack channel {channel}")
                else:
                    self.send_message("Sorry, I couldn't generate the P/L chart at this time (generation failed).")

            except SlackApiError as e:
                logging.error(f"Slack API error during P/L chart command: {e.response['error'] if e.response else str(e)}")
                self.send_message(f"Sorry, a Slack error occurred while uploading the P/L chart: {e.response['error'] if e.response else str(e)}")
            except Exception as e:
                logging.error(f"Unexpected error during P/L chart command: {e}", exc_info=True)
                self.send_message("Sorry, an unexpected error occurred while processing the P/L chart command.")
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                        logging.debug(f"Successfully removed temporary P/L chart file: {temp_file_path}")
                    except OSError as e_remove:
                        logging.error(f"Error removing temporary P/L chart file {temp_file_path}: {e_remove}")
        else:
            response_text = f"Unknown command: '{command_text}'. Try 'start', 'stop', 'status', 'graph', or 'pnl_chart'."
            self.send_message(response_text)


    def start_listening(self):
        """
        Starts listening for Slack RTM events.
        """
        if not self.rtm_client:
            logging.warning("Slack RTM client not initialized. Cannot start listening.")
            return

        logging.info("Slack RTM client starting to listen for events...")
        try:
            for payload in self.rtm_client.start(): # This is a generator
                # logging.debug(f"RTM Payload: {payload}") # Too verbose for normal operation
                if payload.get("type") == "message" and "text" in payload:
                    channel_id = payload.get("channel")
                    user_id = payload.get("user")
                    text = payload.get("text")

                    # Avoid responding to own messages or messages from other bots if bot_id is known
                    if self.bot_id and user_id == self.bot_id:
                        continue

                    # Check if the message is from the configured channel or a direct message
                    # For DMs, channel_id might be different (e.g., starts with 'D')
                    # For simplicity, we'll only process commands from the designated channel_id
                    # or if the message is a direct mention of the bot (e.g. "<@BOT_ID> command")

                    is_mention = self.bot_id and text.startswith(f"<@{self.bot_id}>")
                    is_target_channel = channel_id == self.channel_id

                    if is_target_channel or is_mention:
                        command_text = text
                        if is_mention:
                            # Strip the mention part to get the actual command
                            command_text = text.replace(f"<@{self.bot_id}>", "", 1).strip()

                        if command_text: # Ensure there's a command after potential stripping
                            logging.info(f"Received command '{command_text}' from user {user_id} in channel {channel_id}")
                            self.handle_command(command_text, user_id, channel_id)
        except SlackApiError as e: # Ensure this specific exception is caught if not already generic
            logging.error(f"Slack API Error during RTM listening: {e.response['error'] if e.response else str(e)}")
            if "ratelimited" in str(e).lower():
                logging.warning("RTM client was ratelimited. Pausing before retry...")
                time.sleep(60) # Wait a minute before trying to reconnect (basic backoff)
                self.start_listening() # Attempt to restart listening
            elif "invalid_auth" in str(e).lower() or "not_authed" in str(e).lower():
                logging.error("Slack authentication failed. RTM client will stop.")
                self.rtm_client = None # Stop further attempts if auth fails
            # Add specific handling for other critical errors if needed
            else:
                # For other API errors, retry might be possible after a delay
                logging.info(f"Attempting to recover RTM client after API error: {e}")
                time.sleep(10) # Wait before retrying
                if self.rtm_client: # Only retry if client wasn't disabled by auth failure
                    self.start_listening() # Attempt to restart listening
        except Exception as e: # Catch any other unexpected errors
            logging.error(f"Unexpected exception in Slack RTM event loop: {e}", exc_info=True)
            # Robustness: attempt to restart listening after a delay, unless it's a known fatal error
            time.sleep(10) # Wait before retrying
            if self.rtm_client: # Only retry if client wasn't disabled
                 self.start_listening()
        logging.info("Slack RTM client stopped listening.")


if __name__ == '__main__':
    # This is an example of how to use the SlackInterface
    # You'll need to replace "YOUR_SLACK_TOKEN" and "YOUR_CHANNEL_ID"
    # with your actual Slack token and channel ID.
    # It's also recommended to load these from environment variables or a config file.
    print("SlackInterface example usage (requires valid token and channel_id):")
    # This is an example of how to use the SlackInterface
    # For this example to run, you would need a mock conf object and trading_instance.
    print("SlackInterface example usage (requires valid token and channel_id in a mock config):")
    # class MockConf:
    #     def __init__(self, token, channel_id):
    #         self.slack_token = token
    #         self.slack_channel_id = channel_id
    #
    # class MockTradingInstance:
    #     pass
    #
    # slack_token = "YOUR_SLACK_TOKEN"
    # slack_channel_id = "YOUR_CHANNEL_ID"
    #
    # if slack_token == "YOUR_SLACK_TOKEN" or slack_channel_id == "YOUR_CHANNEL_ID":
    #     print("Please replace YOUR_SLACK_TOKEN and YOUR_CHANNEL_ID with actual values to test.")
    # else:
    #     mock_conf = MockConf(token=slack_token, channel_id=slack_channel_id)
    #     mock_trading = MockTradingInstance()
    #     slack_interface = SlackInterface(conf=mock_conf, trading_instance=mock_trading)
    #     slack_interface.send_message("Hello from the Crypto Trading Bot!")
    #     slack_interface.handle_command("/balance")
    #     slack_interface.start_listening()
