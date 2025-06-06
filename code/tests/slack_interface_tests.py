import unittest
from unittest.mock import patch, Mock, MagicMock
import datetime # For graph test data
import os # For os.path.exists and os.remove mocks

# Assuming crypto_trading is in PYTHONPATH
from crypto_trading.slack_interface import SlackInterface
from slack_sdk.errors import SlackApiError # For testing upload failure
# We will mock Config and Trading rather than importing them directly for most tests
# from crypto_trading.config import Config # Not strictly needed if fully mocked
# from crypto_trading.trading import Trading # Not strictly needed if fully mocked

# Placeholder for where slack_sdk might be if we needed to mock deeper
# import slack_sdk

class TestSlackInterface(unittest.TestCase):

    def setUp(self):
        # Mock configuration
        self.mock_conf = Mock()
        self.mock_conf.slack_token = "xoxb-test-token"
        self.mock_conf.slack_channel_id = "C123TEST"
        self.mock_conf.initial_capital = 1000.0 # Added for graph command
        # In config.py, these are loaded via self.config_dict.get('slack_token'),
        # so they become attributes of the conf object.

        # Mock trading instance
        self.mock_trading_instance = Mock()
        self.mock_trading_instance.is_running = Mock(return_value=False)
        self.mock_trading_instance.profits = Mock(return_value=0.0)
        self.mock_trading_instance.stop = Mock()
        # self.mock_trading_instance.run = Mock() # For 'start' command testing if needed

        # Mock WebClient and RTMClient at the module level where SlackInterface imports them
        # These will be class-level patches usually, or context managers in tests.
        # For now, we'll prepare for their use.


    @patch('crypto_trading.slack_interface.RTMClient')
    @patch('crypto_trading.slack_interface.WebClient')
    def test_initialization_success(self, MockWebClient, MockRTMClient):
        # Setup mock return values for WebClient
        mock_web_client_instance = MockWebClient.return_value
        mock_web_client_instance.auth_test.return_value = {'ok': True, 'user_id': 'U123', 'bot_id': 'B456'}

        # Setup mock return values for RTMClient
        # mock_rtm_client_instance = MockRTMClient.return_value # If RTMClient methods were called in __init__

        slack_interface = SlackInterface(conf=self.mock_conf, trading_instance=self.mock_trading_instance)

        MockWebClient.assert_called_once_with(token="xoxb-test-token")
        mock_web_client_instance.auth_test.assert_called_once()
        MockRTMClient.assert_called_once_with(token="xoxb-test-token")

        self.assertIsNotNone(slack_interface.client, "Client should be initialized on success")
        self.assertIsNotNone(slack_interface.rtm_client, "RTMClient should be initialized on success")
        self.assertEqual(slack_interface.bot_id, "B456")


    @patch('crypto_trading.slack_interface.RTMClient')
    @patch('crypto_trading.slack_interface.WebClient')
    def test_initialization_no_token(self, MockWebClient, MockRTMClient):
        self.mock_conf.slack_token = None
        slack_interface = SlackInterface(conf=self.mock_conf, trading_instance=self.mock_trading_instance)

        MockWebClient.assert_not_called()
        MockRTMClient.assert_not_called()
        self.assertIsNone(slack_interface.client, "Client should be None if token is missing")
        self.assertIsNone(slack_interface.rtm_client, "RTMClient should be None if token is missing")


    @patch('crypto_trading.slack_interface.RTMClient')
    @patch('crypto_trading.slack_interface.WebClient')
    def test_initialization_auth_failure(self, MockWebClient, MockRTMClient):
        mock_web_client_instance = MockWebClient.return_value
        mock_web_client_instance.auth_test.return_value = {'ok': False, 'error': 'auth_failed'}

        slack_interface = SlackInterface(conf=self.mock_conf, trading_instance=self.mock_trading_instance)

        MockWebClient.assert_called_once_with(token="xoxb-test-token")
        mock_web_client_instance.auth_test.assert_called_once()
        # RTMClient IS called before auth_test, but slack_interface.rtm_client is set to None if auth fails
        MockRTMClient.assert_called_once_with(token="xoxb-test-token")

        self.assertIsNone(slack_interface.client, "Client should be None on auth failure")
        # In the current implementation, RTMClient is initialized before auth_test.
        # It might be better to set both to None if auth_test fails.
        # The current code sets both client and rtm_client to None if auth_test fails or an exception occurs.
        self.assertIsNone(slack_interface.rtm_client, "RTMClient should be None on auth failure")


    @patch('crypto_trading.slack_interface.RTMClient')
    @patch('crypto_trading.slack_interface.WebClient')
    def test_send_message_success(self, MockWebClient, MockRTMClient):
        # Ensure interface is initialized successfully for this test
        mock_web_client_instance = MockWebClient.return_value
        mock_web_client_instance.auth_test.return_value = {'ok': True, 'user_id': 'U123', 'bot_id': 'B456'}

        slack_interface = SlackInterface(conf=self.mock_conf, trading_instance=self.mock_trading_instance)
        self.assertIsNotNone(slack_interface.client) # Pre-condition

        slack_interface.send_message("test message")
        mock_web_client_instance.chat_postMessage.assert_called_once_with(
            channel=self.mock_conf.slack_channel_id, text="test message"
        )

    @patch('crypto_trading.slack_interface.RTMClient')
    @patch('crypto_trading.slack_interface.WebClient')
    def test_send_message_client_disabled(self, MockWebClient, MockRTMClient):
        self.mock_conf.slack_token = None # Disable client
        slack_interface = SlackInterface(conf=self.mock_conf, trading_instance=self.mock_trading_instance)
        self.assertIsNone(slack_interface.client) # Pre-condition

        # Get the underlying mock for the WebClient instance to check calls
        # Since client is None, no instance is stored on slack_interface.client
        # We need to check the original MockWebClient's return_value if it was ever created.
        # In this case (no token), MockWebClient() is not called.

        slack_interface.send_message("test message")
        MockWebClient.return_value.chat_postMessage.assert_not_called()


    @patch('crypto_trading.slack_interface.SlackInterface.send_message') # Mock send_message
    @patch('crypto_trading.slack_interface.RTMClient')
    @patch('crypto_trading.slack_interface.WebClient')
    def test_handle_command_start_running(self, MockWebClient, MockRTMClient, mock_send_message_method):
        mock_web_client_instance = MockWebClient.return_value
        mock_web_client_instance.auth_test.return_value = {'ok': True, 'bot_id': 'B456'}

        self.mock_trading_instance.is_running.return_value = True # Bot is already running
        slack_interface = SlackInterface(conf=self.mock_conf, trading_instance=self.mock_trading_instance)

        slack_interface.handle_command("start", "user1", "channel1")
        # self.mock_trading_instance.run.assert_not_called() # or assert_called_once if it's idempotent
        mock_send_message_method.assert_called_once_with("Trading bot is already running.")

    @patch('crypto_trading.slack_interface.SlackInterface.send_message')
    @patch('crypto_trading.slack_interface.RTMClient')
    @patch('crypto_trading.slack_interface.WebClient')
    def test_handle_command_start_stopped_cannot_restart(self, MockWebClient, MockRTMClient, mock_send_message_method):
        mock_web_client_instance = MockWebClient.return_value
        mock_web_client_instance.auth_test.return_value = {'ok': True, 'bot_id': 'B456'}

        self.mock_trading_instance.is_running.return_value = False # Bot is stopped
        slack_interface = SlackInterface(conf=self.mock_conf, trading_instance=self.mock_trading_instance)

        slack_interface.handle_command("start", "user1", "channel1")
        # self.mock_trading_instance.run.assert_not_called() # Current design limitation
        mock_send_message_method.assert_called_once_with(
            "Trading bot is currently stopped. A 'start' command from Slack cannot restart a fully exited trading loop without a redesign. Please restart the application if needed."
        )

    @patch('crypto_trading.slack_interface.SlackInterface.send_message')
    @patch('crypto_trading.slack_interface.RTMClient')
    @patch('crypto_trading.slack_interface.WebClient')
    def test_handle_command_stop(self, MockWebClient, MockRTMClient, mock_send_message_method):
        mock_web_client_instance = MockWebClient.return_value
        mock_web_client_instance.auth_test.return_value = {'ok': True, 'bot_id': 'B456'}
        slack_interface = SlackInterface(conf=self.mock_conf, trading_instance=self.mock_trading_instance)

        slack_interface.handle_command("stop", "user1", "channel1")
        self.mock_trading_instance.stop.assert_called_once()
        mock_send_message_method.assert_called_once_with("Trading bot stop signal sent.")

    @patch('crypto_trading.slack_interface.SlackInterface.send_message')
    @patch('crypto_trading.slack_interface.RTMClient')
    @patch('crypto_trading.slack_interface.WebClient')
    def test_handle_command_status_running(self, MockWebClient, MockRTMClient, mock_send_message_method):
        mock_web_client_instance = MockWebClient.return_value
        mock_web_client_instance.auth_test.return_value = {'ok': True, 'bot_id': 'B456'}
        self.mock_trading_instance.is_running.return_value = True
        self.mock_trading_instance.profits.return_value = 123.45
        slack_interface = SlackInterface(conf=self.mock_conf, trading_instance=self.mock_trading_instance)

        slack_interface.handle_command("status", "user1", "channel1")
        mock_send_message_method.assert_called_once_with("Bot status: Running. Current profits: 123.45")

    @patch('crypto_trading.slack_interface.SlackInterface.send_message')
    @patch('crypto_trading.slack_interface.RTMClient')
    @patch('crypto_trading.slack_interface.WebClient')
    def test_handle_command_status_stopped(self, MockWebClient, MockRTMClient, mock_send_message_method):
        mock_web_client_instance = MockWebClient.return_value
        mock_web_client_instance.auth_test.return_value = {'ok': True, 'bot_id': 'B456'}
        self.mock_trading_instance.is_running.return_value = False
        self.mock_trading_instance.profits.return_value = 0.0
        slack_interface = SlackInterface(conf=self.mock_conf, trading_instance=self.mock_trading_instance)

        slack_interface.handle_command("status", "user1", "channel1")
        mock_send_message_method.assert_called_once_with("Bot status: Stopped. Current profits: 0.0")

    @patch('crypto_trading.slack_interface.SlackInterface.send_message')
    @patch('crypto_trading.slack_interface.RTMClient')
    @patch('crypto_trading.slack_interface.WebClient')
    def test_handle_command_unknown(self, MockWebClient, MockRTMClient, mock_send_message_method):
        mock_web_client_instance = MockWebClient.return_value
        mock_web_client_instance.auth_test.return_value = {'ok': True, 'bot_id': 'B456'}
        # This instance of slack_interface will be used for most command tests
        # It's important that it's initialized correctly for client to be not None
        self.slack_interface_for_commands = SlackInterface(conf=self.mock_conf, trading_instance=self.mock_trading_instance)
        self.assertIsNotNone(self.slack_interface_for_commands.client) # Ensure client is active for these tests

        self.slack_interface_for_commands.handle_command("unknown_command", "user1", "channel1")
        mock_send_message_method.assert_called_once_with(
            "Unknown command: 'unknown_command'. Try 'start', 'stop', 'status', 'graph', or 'pnl_chart'."
        )


    # --- Tests for 'graph' command ---
    @patch('crypto_trading.slack_interface.os.remove')
    @patch('crypto_trading.slack_interface.tempfile.NamedTemporaryFile')
    @patch('crypto_trading.slack_interface.generate_portfolio_graph')
    @patch('crypto_trading.slack_interface.model.get_portfolio_value_history')
    @patch('crypto_trading.slack_interface.WebClient') # To get the client instance for files_upload_v2
    @patch('crypto_trading.slack_interface.RTMClient') # Standard patch
    def test_handle_command_graph_success(self, MockRTMClient, MockWebClient, mock_get_history,
                                          mock_generate_graph, mock_NamedTemporaryFile, mock_os_remove):
        # Setup WebClient mock instance for this test
        mock_web_client_instance = MockWebClient.return_value
        mock_web_client_instance.auth_test.return_value = {'ok': True, 'bot_id': 'B456'}

        # Initialize SlackInterface here to use the method-scoped MockWebClient
        slack_interface = SlackInterface(conf=self.mock_conf, trading_instance=self.mock_trading_instance)
        self.assertIsNotNone(slack_interface.client) # Ensure client is active

        sample_data_points = [(datetime.datetime.now(), 1000.0), (datetime.datetime.now(), 1050.0)]
        mock_get_history.return_value = sample_data_points

        mock_graph_file_path = "dummy_temp_graph.png"
        mock_generate_graph.return_value = mock_graph_file_path

        # Mock tempfile.NamedTemporaryFile
        mock_temp_file_obj = Mock()
        mock_temp_file_obj.name = mock_graph_file_path
        mock_named_temp_file_cm = Mock() # Context Manager
        mock_named_temp_file_cm.__enter__.return_value = mock_temp_file_obj
        mock_named_temp_file_cm.__exit__.return_value = None
        mock_NamedTemporaryFile.return_value = mock_named_temp_file_cm

        # Mock os.path.exists for the finally block
        with patch('crypto_trading.slack_interface.os.path.exists', return_value=True) as mock_os_exists:
            slack_interface.handle_command("graph", "user1", "channel1")

        mock_get_history.assert_called_once_with(self.mock_conf.initial_capital)
        mock_NamedTemporaryFile.assert_called_once_with(suffix=".png", delete=False)
        mock_generate_graph.assert_called_once_with(sample_data_points, mock_graph_file_path)

        # Use the client instance from the initialized slack_interface
        slack_interface.client.files_upload_v2.assert_called_once_with(
            channel="channel1",
            file=mock_graph_file_path,
            title="Portfolio Value Over Time",
            initial_comment="Here is the portfolio value graph:"
        )
        mock_os_exists.assert_called_once_with(mock_graph_file_path)
        mock_os_remove.assert_called_once_with(mock_graph_file_path)


    @patch.object(SlackInterface, 'send_message')
    @patch('crypto_trading.slack_interface.generate_portfolio_graph') # To ensure it's not called
    @patch('crypto_trading.slack_interface.model.get_portfolio_value_history')
    @patch('crypto_trading.slack_interface.WebClient')
    @patch('crypto_trading.slack_interface.RTMClient')
    def test_handle_command_graph_insufficient_data(self, MockRTMClient, MockWebClient, mock_get_history,
                                                    mock_generate_graph, mock_send_message_method):
        mock_web_client_instance = MockWebClient.return_value
        mock_web_client_instance.auth_test.return_value = {'ok': True, 'bot_id': 'B456'}
        slack_interface = SlackInterface(conf=self.mock_conf, trading_instance=self.mock_trading_instance)

        mock_get_history.return_value = [(datetime.datetime.now(), 1000.0)] # Only one point

        slack_interface.handle_command("graph", "user1", "channel1")

        mock_get_history.assert_called_once_with(self.mock_conf.initial_capital)
        mock_send_message_method.assert_called_once_with("Not enough data to generate a graph yet.")
        mock_generate_graph.assert_not_called()
        slack_interface.client.files_upload_v2.assert_not_called()


    @patch('crypto_trading.slack_interface.os.remove')
    @patch('crypto_trading.slack_interface.tempfile.NamedTemporaryFile')
    @patch('crypto_trading.slack_interface.generate_portfolio_graph', return_value=None) # Mock graph generation failure
    @patch('crypto_trading.slack_interface.model.get_portfolio_value_history')
    @patch.object(SlackInterface, 'send_message') # Mocking send_message directly on the class
    @patch('crypto_trading.slack_interface.WebClient')
    @patch('crypto_trading.slack_interface.RTMClient')
    def test_handle_command_graph_generation_fails(self, MockRTMClient, MockWebClient, mock_send_message_method,
                                                 mock_get_history, mock_generate_graph_fails,
                                                 mock_NamedTemporaryFile, mock_os_remove):
        mock_web_client_instance = MockWebClient.return_value
        mock_web_client_instance.auth_test.return_value = {'ok': True, 'bot_id': 'B456'}
        slack_interface = SlackInterface(conf=self.mock_conf, trading_instance=self.mock_trading_instance)

        sample_data_points = [(datetime.datetime.now(), 1000.0), (datetime.datetime.now(), 1050.0)]
        mock_get_history.return_value = sample_data_points

        mock_graph_file_path = "dummy_temp_graph.png"
        mock_temp_file_obj = Mock()
        mock_temp_file_obj.name = mock_graph_file_path
        mock_named_temp_file_cm = Mock()
        mock_named_temp_file_cm.__enter__.return_value = mock_temp_file_obj
        mock_named_temp_file_cm.__exit__.return_value = None
        mock_NamedTemporaryFile.return_value = mock_named_temp_file_cm

        with patch('crypto_trading.slack_interface.os.path.exists', return_value=True) as mock_os_exists:
            slack_interface.handle_command("graph", "user1", "channel1")

        mock_get_history.assert_called_once_with(self.mock_conf.initial_capital)
        mock_generate_graph_fails.assert_called_once()
        mock_send_message_method.assert_called_once_with(
            "Sorry, I couldn't generate the portfolio graph at this time (generation failed)."
        )
        slack_interface.client.files_upload_v2.assert_not_called()
        mock_os_exists.assert_called_once_with(mock_graph_file_path)
        mock_os_remove.assert_called_once_with(mock_graph_file_path)


    @patch('crypto_trading.slack_interface.os.remove')
    @patch('crypto_trading.slack_interface.tempfile.NamedTemporaryFile')
    @patch('crypto_trading.slack_interface.generate_portfolio_graph')
    @patch('crypto_trading.slack_interface.model.get_portfolio_value_history')
    @patch.object(SlackInterface, 'send_message')
    @patch('crypto_trading.slack_interface.WebClient')
    @patch('crypto_trading.slack_interface.RTMClient')
    def test_handle_command_graph_slack_upload_fails(self, MockRTMClient, MockWebClient, mock_send_message_method,
                                                    mock_get_history, mock_generate_graph,
                                                    mock_NamedTemporaryFile, mock_os_remove):
        mock_web_client_instance = MockWebClient.return_value
        mock_web_client_instance.auth_test.return_value = {'ok': True, 'bot_id': 'B456'}

        # This specific instance of WebClient's method needs to raise an error
        mock_web_client_instance.files_upload_v2.side_effect = SlackApiError(
            message="Upload failed", response={"error": "upload_error", "ok": False}
        )

        # Re-initialize slack_interface AFTER setting up the WebClient mock's side_effect for files_upload_v2
        # if WebClient is patched at method level.
        # If WebClient is patched at class level, this should be fine.
        # For safety, we can ensure slack_interface.client is this mocked instance.
        slack_interface = SlackInterface(conf=self.mock_conf, trading_instance=self.mock_trading_instance)
        slack_interface.client = mock_web_client_instance # Ensure it uses the specifically mocked client

        sample_data_points = [(datetime.datetime.now(), 1000.0), (datetime.datetime.now(), 1050.0)]
        mock_get_history.return_value = sample_data_points

        mock_graph_file_path = "dummy_temp_graph.png"
        mock_generate_graph.return_value = mock_graph_file_path

        mock_temp_file_obj = Mock()
        mock_temp_file_obj.name = mock_graph_file_path
        mock_named_temp_file_cm = Mock()
        mock_named_temp_file_cm.__enter__.return_value = mock_temp_file_obj
        mock_named_temp_file_cm.__exit__.return_value = None
        mock_NamedTemporaryFile.return_value = mock_named_temp_file_cm

        with patch('crypto_trading.slack_interface.os.path.exists', return_value=True) as mock_os_exists:
            slack_interface.handle_command("graph", "user1", "channel1")

        mock_get_history.assert_called_once_with(self.mock_conf.initial_capital)
        mock_generate_graph.assert_called_once()
        slack_interface.client.files_upload_v2.assert_called_once() # It was called
        mock_send_message_method.assert_called_once_with(
            "Sorry, a Slack error occurred while uploading the graph: upload_error"
        )
        mock_os_exists.assert_called_once_with(mock_graph_file_path)
        mock_os_remove.assert_called_once_with(mock_graph_file_path)

    # Test for start_listening is more complex and will be simplified

    @patch('crypto_trading.slack_interface.time') # Mock time.sleep
    @patch.object(SlackInterface, 'handle_command') # Mock the instance method
    @patch('crypto_trading.slack_interface.RTMClient')
    @patch('crypto_trading.slack_interface.WebClient')
    def test_start_listening_event_processing_and_ignore_own_message(
            self, MockWebClient, MockRTMClient, mock_handle_command_method, mock_time):

        mock_web_client_instance = MockWebClient.return_value
        mock_web_client_instance.auth_test.return_value = {'ok': True, 'bot_id': 'B456'} # Bot's own ID

        mock_rtm_instance = MockRTMClient.return_value

        # Simulate two messages: one from a user, one from the bot itself
        test_payload_user = {
            'type': 'message', 'text': 'status', 'user': 'UUSER1',
            'channel': self.mock_conf.slack_channel_id, 'ts': '12345.678'
        }
        test_payload_bot_own_message = {
            'type': 'message', 'text': 'some response', 'user': 'B456', # Message from bot
            'channel': self.mock_conf.slack_channel_id, 'ts': '12345.679'
        }
        test_payload_other_channel_no_mention = {
             'type': 'message', 'text': 'status', 'user': 'UUSER2',
             'channel': 'COTHERCHANNEL', 'ts': '12345.680'
        }
        test_payload_other_channel_with_mention = {
            'type': 'message', 'text': '<@B456> status', 'user': 'UUSER3',
            'channel': 'COTHERCHANNEL2', 'ts': '12345.681'
        }
        # Simulate RTMClient.start() yielding these payloads and then an error to break the loop for the test
        mock_rtm_instance.start.return_value = [
            test_payload_user,
            test_payload_bot_own_message,
            test_payload_other_channel_no_mention,
            test_payload_other_channel_with_mention,
            Exception("Stop loop for test") # To break out of the while True
        ]

        slack_interface = SlackInterface(conf=self.mock_conf, trading_instance=self.mock_trading_instance)

        # Call start_listening. It will loop until Exception("Stop loop for test")
        with self.assertRaisesRegex(Exception, "Stop loop for test"):
            slack_interface.start_listening()

        # Assert handle_command was called for the user's message in target channel
        mock_handle_command_method.assert_any_call(
            'status', 'UUSER1', self.mock_conf.slack_channel_id
        )
        # Assert handle_command was called for the user's message in other channel with mention
        mock_handle_command_method.assert_any_call(
            'status', 'UUSER3', 'COTHERCHANNEL2' # text is stripped of mention
        )

        # Check total calls to handle_command to ensure bot's own message and other_channel_no_mention were ignored
        self.assertEqual(mock_handle_command_method.call_count, 2)


if __name__ == '__main__':
    unittest.main()


    # --- Tests for 'pnl_chart' command ---

    def create_mock_sqlobject_trade(self, sell_date_time, profit):
        trade = Mock()
        trade.sell_date_time = sell_date_time
        trade.profit = profit
        return trade

    @patch('crypto_trading.slack_interface.os.remove')
    @patch('crypto_trading.slack_interface.tempfile.NamedTemporaryFile')
    @patch('crypto_trading.slack_interface.generate_pnl_per_trade_graph')
    @patch('crypto_trading.slack_interface.model.Trading.select')
    @patch('crypto_trading.slack_interface.WebClient')
    @patch('crypto_trading.slack_interface.RTMClient')
    def test_handle_command_pnl_chart_success(self, MockRTMClient, MockWebClient, mock_trading_select,
                                             mock_generate_pnl_graph, mock_NamedTemporaryFile, mock_os_remove):
        mock_web_client_instance = MockWebClient.return_value
        mock_web_client_instance.auth_test.return_value = {'ok': True, 'bot_id': 'B456'}
        slack_interface = SlackInterface(conf=self.mock_conf, trading_instance=self.mock_trading_instance)

        mock_trade1_sell_time = datetime.datetime(2023, 1, 1, 11, 0, 0)
        mock_trade1 = self.create_mock_sqlobject_trade(mock_trade1_sell_time, 50.0)
        mock_trade2_sell_time = datetime.datetime(2023, 1, 2, 11, 0, 0)
        mock_trade2 = self.create_mock_sqlobject_trade(mock_trade2_sell_time, -20.0)

        mock_trading_select.return_value = [mock_trade1, mock_trade2]

        mock_graph_file_path = "dummy_pnl_chart.png"
        mock_generate_pnl_graph.return_value = mock_graph_file_path

        mock_temp_file_obj = Mock()
        mock_temp_file_obj.name = mock_graph_file_path
        mock_named_temp_file_cm = Mock()
        mock_named_temp_file_cm.__enter__.return_value = mock_temp_file_obj
        mock_NamedTemporaryFile.return_value = mock_named_temp_file_cm

        with patch('crypto_trading.slack_interface.os.path.exists', return_value=True) as mock_os_exists:
            slack_interface.handle_command("pnl_chart", "user1", "channel1")

        mock_trading_select.assert_called_once() # Basic check, can be more specific on query

        expected_trades_data = [
            {'label': '2023-01-01 11:00', 'profit': 50.0},
            {'label': '2023-01-02 11:00', 'profit': -20.0}
        ]
        mock_generate_pnl_graph.assert_called_once_with(expected_trades_data, mock_graph_file_path)

        slack_interface.client.files_upload_v2.assert_called_once_with(
            channel="channel1",
            file=mock_graph_file_path,
            title="Profit/Loss per Trade",
            initial_comment="Here is the P/L chart for completed trades:"
        )
        mock_os_exists.assert_called_once_with(mock_graph_file_path)
        mock_os_remove.assert_called_once_with(mock_graph_file_path)

    @patch.object(SlackInterface, 'send_message')
    @patch('crypto_trading.slack_interface.generate_pnl_per_trade_graph')
    @patch('crypto_trading.slack_interface.model.Trading.select', return_value=[]) # No trades
    @patch('crypto_trading.slack_interface.WebClient')
    @patch('crypto_trading.slack_interface.RTMClient')
    def test_handle_command_pnl_chart_no_trades(self, MockRTMClient, MockWebClient, mock_trading_select,
                                               mock_generate_pnl_graph, mock_send_message_method):
        mock_web_client_instance = MockWebClient.return_value
        mock_web_client_instance.auth_test.return_value = {'ok': True, 'bot_id': 'B456'}
        slack_interface = SlackInterface(conf=self.mock_conf, trading_instance=self.mock_trading_instance)

        slack_interface.handle_command("pnl_chart", "user1", "channel1")

        mock_trading_select.assert_called_once()
        mock_send_message_method.assert_called_once_with("No completed trades found to generate a P/L chart.")
        mock_generate_pnl_graph.assert_not_called()


    @patch('crypto_trading.slack_interface.os.remove')
    @patch('crypto_trading.slack_interface.tempfile.NamedTemporaryFile')
    @patch('crypto_trading.slack_interface.generate_pnl_per_trade_graph', return_value=None) # Gen fails
    @patch('crypto_trading.slack_interface.model.Trading.select')
    @patch.object(SlackInterface, 'send_message')
    @patch('crypto_trading.slack_interface.WebClient')
    @patch('crypto_trading.slack_interface.RTMClient')
    def test_handle_command_pnl_chart_generation_fails(self, MockRTMClient, MockWebClient, mock_send_message,
                                                       mock_trading_select, mock_generate_pnl_graph_fails,
                                                       mock_NamedTemporaryFile, mock_os_remove):
        mock_web_client_instance = MockWebClient.return_value
        mock_web_client_instance.auth_test.return_value = {'ok': True, 'bot_id': 'B456'}
        slack_interface = SlackInterface(conf=self.mock_conf, trading_instance=self.mock_trading_instance)

        mock_trade1 = self.create_mock_sqlobject_trade(datetime.datetime.now(), 50.0)
        mock_trading_select.return_value = [mock_trade1] # Some data

        mock_graph_file_path = "dummy_pnl_chart.png"
        mock_temp_file_obj = Mock()
        mock_temp_file_obj.name = mock_graph_file_path
        mock_named_temp_file_cm = Mock()
        mock_named_temp_file_cm.__enter__.return_value = mock_temp_file_obj
        mock_NamedTemporaryFile.return_value = mock_named_temp_file_cm

        with patch('crypto_trading.slack_interface.os.path.exists', return_value=True) as mock_os_exists:
            slack_interface.handle_command("pnl_chart", "user1", "channel1")

        mock_generate_pnl_graph_fails.assert_called_once()
        mock_send_message.assert_called_once_with(
            "Sorry, I couldn't generate the P/L chart at this time (generation failed)."
        )
        mock_os_exists.assert_called_once_with(mock_graph_file_path)
        mock_os_remove.assert_called_once_with(mock_graph_file_path)


    @patch('crypto_trading.slack_interface.os.remove')
    @patch('crypto_trading.slack_interface.tempfile.NamedTemporaryFile')
    @patch('crypto_trading.slack_interface.generate_pnl_per_trade_graph')
    @patch('crypto_trading.slack_interface.model.Trading.select')
    @patch.object(SlackInterface, 'send_message')
    @patch('crypto_trading.slack_interface.WebClient')
    @patch('crypto_trading.slack_interface.RTMClient')
    def test_handle_command_pnl_chart_slack_upload_fails(self, MockRTMClient, MockWebClient, mock_send_message,
                                                        mock_trading_select, mock_generate_pnl_graph,
                                                        mock_NamedTemporaryFile, mock_os_remove):
        mock_web_client_instance = MockWebClient.return_value
        mock_web_client_instance.auth_test.return_value = {'ok': True, 'bot_id': 'B456'}
        mock_web_client_instance.files_upload_v2.side_effect = SlackApiError(
            message="PNL Upload failed", response={"error": "pnl_upload_error", "ok": False}
        )
        slack_interface = SlackInterface(conf=self.mock_conf, trading_instance=self.mock_trading_instance)
        slack_interface.client = mock_web_client_instance # Ensure our mock instance is used

        mock_trade1 = self.create_mock_sqlobject_trade(datetime.datetime.now(), 50.0)
        mock_trading_select.return_value = [mock_trade1]

        mock_graph_file_path = "dummy_pnl_chart.png"
        mock_generate_pnl_graph.return_value = mock_graph_file_path

        mock_temp_file_obj = Mock()
        mock_temp_file_obj.name = mock_graph_file_path
        mock_named_temp_file_cm = Mock()
        mock_named_temp_file_cm.__enter__.return_value = mock_temp_file_obj
        mock_NamedTemporaryFile.return_value = mock_named_temp_file_cm

        with patch('crypto_trading.slack_interface.os.path.exists', return_value=True) as mock_os_exists:
            slack_interface.handle_command("pnl_chart", "user1", "channel1")

        mock_generate_pnl_graph.assert_called_once()
        slack_interface.client.files_upload_v2.assert_called_once()
        mock_send_message.assert_called_once_with(
            "Sorry, a Slack error occurred while uploading the P/L chart: pnl_upload_error"
        )
        mock_os_exists.assert_called_once_with(mock_graph_file_path)
        mock_os_remove.assert_called_once_with(mock_graph_file_path)
