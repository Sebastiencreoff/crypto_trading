import unittest
from unittest.mock import patch, Mock, MagicMock, call
import datetime
import tempfile
import os # For os.path.exists and os.remove

# Assuming 'code' is project root in PYTHONPATH, or sys.path is adjusted
from crypto_trading.slack.slack_interface import SlackCommandHandler
from crypto_trading.task_manager import TaskManager # Mocked, but import for type hinting if desired
from config_management.schemas import SlackConfig # For creating mock config
from slack_sdk.errors import SlackApiError

# Paths for patching plotting and tempfile where they are used in slack_interface.py
SLACK_INTERFACE_MODULE_PATH = "crypto_trading.slack.slack_interface"

class TestSlackCommandHandler(unittest.TestCase):

    def setUp(self):
        # 1. Mock Configuration (SlackConfig)
        self.mock_slack_config = SlackConfig(
            bot_token="xoxb-test-token",
            default_channel_id="C123TEST",
            admin_user_ids=["UADMIN1"], # Example, not directly used by current SCH commands
            task_docker_image="test-task-image:latest" # Example
        )

        # 2. Mock TaskManager
        self.mock_task_manager = Mock(spec=TaskManager)
        self.mock_task_manager.create_task = Mock(return_value="task_uuid_123")
        self.mock_task_manager.stop_task = Mock(return_value=True)
        self.mock_task_manager.get_task_status = Mock(return_value="running")
        self.mock_task_manager.list_tasks = Mock(return_value={"task_1": "running", "task_2": "completed"})
        self.mock_task_manager.get_task_results = Mock(return_value="Sample task logs or results.")

        # 3. Patch slack_sdk clients used by SlackCommandHandler
        # These patches will apply to all tests in this class.
        # We get the mock instances in each test or here if setup is complex.
        self.web_client_patcher = patch(f'{SLACK_INTERFACE_MODULE_PATH}.WebClient')
        self.rtm_client_patcher = patch(f'{SLACK_INTERFACE_MODULE_PATH}.RTMClient')

        self.MockWebClient = self.web_client_patcher.start()
        self.MockRTMClient = self.rtm_client_patcher.start()

        # Configure the mock WebClient instance that will be returned
        self.mock_web_client_instance = self.MockWebClient.return_value
        self.mock_web_client_instance.auth_test.return_value = {'ok': True, 'user': 'test_bot_user'}
        self.mock_web_client_instance.chat_postMessage = Mock(return_value={'ok': True, 'message': {'ts': '123.456'}})
        self.mock_web_client_instance.files_upload_v2 = Mock(return_value={'ok': True, 'files': []})

        # Configure the mock RTMClient instance
        self.mock_rtm_client_instance = self.MockRTMClient.return_value
        # RTMClient.start() is blocking, so we don't mock its direct output here unless testing start_listening loop.
        # For command tests, we call handler.handle_command directly.

        # 4. Initialize SlackCommandHandler
        # The 'conf' passed to SlackCommandHandler is our mock_slack_config
        self.handler = SlackCommandHandler(conf=self.mock_slack_config, task_manager=self.mock_task_manager)

        # For convenience, mock _send_message to prevent actual Slack calls during most command tests
        # We can then assert its calls.
        # self.handler._send_message = Mock()
        # Replaced by asserting self.mock_web_client_instance.chat_postMessage calls

    def tearDown(self):
        self.web_client_patcher.stop()
        self.rtm_client_patcher.stop()

    # --- Initialization Tests ---
    def test_initialization_success(self):
        self.MockWebClient.assert_called_once_with(token="xoxb-test-token")
        self.mock_web_client_instance.auth_test.assert_called_once()
        self.MockRTMClient.assert_called_once_with(token="xoxb-test-token")
        self.assertTrue(self.handler.is_initialized())
        self.assertEqual(self.handler.task_docker_image, "test-task-image:latest")


    @patch.dict(os.environ, {"SLACK_BOT_TOKEN": "env-token"}, clear=True)
    def test_initialization_success_from_env_token(self):
        # Test if token from env is used when not in config
        mock_conf_no_token = SlackConfig(bot_token=None, default_channel_id="C123")

        # Re-patch WebClient for this specific scenario if its instance is modified in setUp
        # For this test, we create a new handler
        mock_web_client_inst = self.MockWebClient.return_value # Use the already patched one from setUp
        mock_web_client_inst.auth_test.return_value = {'ok': True, 'user': 'test_bot_user_env'}

        handler_env = SlackCommandHandler(conf=mock_conf_no_token, task_manager=self.mock_task_manager)

        self.MockWebClient.assert_called_with(token="env-token") # Check last call or specific call
        mock_web_client_inst.auth_test.assert_called() # Check last call
        self.MockRTMClient.assert_called_with(token="env-token")
        self.assertTrue(handler_env.is_initialized())


    def test_initialization_no_token_anywhere(self):
        # Ensure env doesn't have the token for this test
        with patch.dict(os.environ, {}, clear=True):
            mock_conf_no_token = SlackConfig(bot_token=None, default_channel_id="C123")
            # Reset call counts for global mocks if they are re-used across handlers
            self.MockWebClient.reset_mock()
            self.MockRTMClient.reset_mock()

            handler_no_token = SlackCommandHandler(conf=mock_conf_no_token, task_manager=self.mock_task_manager)

            self.MockWebClient.assert_not_called() # Because token discovery fails early
            self.MockRTMClient.assert_not_called()
            self.assertFalse(handler_no_token.is_initialized())

    def test_initialization_auth_failure(self):
        self.mock_web_client_instance.auth_test.return_value = {'ok': False, 'error': 'auth_failed'}
        # Re-initialize handler to pick up the changed auth_test mock
        handler_auth_fail = SlackCommandHandler(conf=self.mock_slack_config, task_manager=self.mock_task_manager)

        self.MockWebClient.assert_called_with(token="xoxb-test-token")
        self.mock_web_client_instance.auth_test.assert_called()
        self.MockRTMClient.assert_not_called() # RTMClient init is skipped if WebClient auth fails
        self.assertFalse(handler_auth_fail.is_initialized())

    # --- Command Tests ---
    # We will call self.handler.handle_command(command_string, user, channel)
    # And assert calls to self.mock_web_client_instance.chat_postMessage for text responses
    # or self.mock_web_client_instance.files_upload_v2 for file uploads.

    def test_handle_command_help(self):
        self.handler.handle_command("help", "UUSER1", "CCHANNEL1")
        expected_help_text = self.handler._handle_help_command() # Get the exact text
        self.mock_web_client_instance.chat_postMessage.assert_called_once_with(
            channel="CCHANNEL1", text=expected_help_text
        )

    def test_handle_command_start(self):
        self.handler.handle_command("start", "UUSER1", "CCHANNEL1")
        self.mock_task_manager.create_task.assert_called_once_with(self.handler.default_task_config)
        self.mock_web_client_instance.chat_postMessage.assert_called_once_with(
            channel="CCHANNEL1", text="Trading task started successfully. Task ID: `task_uuid_123`"
        )

    def test_handle_command_start_failure(self):
        self.mock_task_manager.create_task.return_value = None # Simulate failure
        self.handler.handle_command("start", "UUSER1", "CCHANNEL1")
        self.mock_task_manager.create_task.assert_called_once_with(self.handler.default_task_config)
        self.mock_web_client_instance.chat_postMessage.assert_called_once_with(
            channel="CCHANNEL1", text="Error: Failed to start trading task. Check logs for details."
        )

    def test_handle_command_stop_success(self):
        self.handler.handle_command("stop task_abc", "UUSER1", "CCHANNEL1")
        self.mock_task_manager.stop_task.assert_called_once_with("task_abc")
        self.mock_web_client_instance.chat_postMessage.assert_called_once_with(
            channel="CCHANNEL1", text="Stop signal sent for task `task_abc`. It may take a moment to terminate."
        )

    def test_handle_command_stop_failure(self):
        self.mock_task_manager.stop_task.return_value = False # Simulate failure
        self.handler.handle_command("stop task_abc", "UUSER1", "CCHANNEL1")
        self.mock_task_manager.stop_task.assert_called_once_with("task_abc")
        self.mock_web_client_instance.chat_postMessage.assert_called_once_with(
            channel="CCHANNEL1", text="Error: Failed to stop task `task_abc`. It might have already completed or does not exist. Check logs."
        )

    def test_handle_command_stop_no_task_id(self):
        self.handler.handle_command("stop", "UUSER1", "CCHANNEL1")
        self.mock_task_manager.stop_task.assert_not_called()
        self.mock_web_client_instance.chat_postMessage.assert_called_once_with(
            channel="CCHANNEL1", text="Error: `task_id` is required for the stop command. Usage: `!crypto stop <task_id>`"
        )

    def test_handle_command_status_specific_task(self):
        self.mock_task_manager.get_task_status.return_value = "completed"
        self.handler.handle_command("status task_xyz", "UUSER1", "CCHANNEL1")
        self.mock_task_manager.get_task_status.assert_called_once_with("task_xyz")
        self.mock_web_client_instance.chat_postMessage.assert_called_once_with(
            channel="CCHANNEL1", text="Status for task `task_xyz`: completed"
        )

    def test_handle_command_status_specific_task_not_found(self):
        self.mock_task_manager.get_task_status.return_value = None
        self.handler.handle_command("status task_nonexistent", "UUSER1", "CCHANNEL1")
        self.mock_task_manager.get_task_status.assert_called_once_with("task_nonexistent")
        self.mock_web_client_instance.chat_postMessage.assert_called_once_with(
            channel="CCHANNEL1", text="Error: Could not retrieve status for task `task_nonexistent`. It may not exist or an error occurred."
        )

    def test_handle_command_status_list_all(self):
        self.mock_task_manager.list_tasks.return_value = {"task_1": "running", "task_2": "stopped"}
        self.handler.handle_command("status", "UUSER1", "CCHANNEL1")
        self.mock_task_manager.list_tasks.assert_called_once()
        expected_message = "Current tasks and their statuses:\n- Task `task_1`: running\n- Task `task_2`: stopped"
        self.mock_web_client_instance.chat_postMessage.assert_called_once_with(
            channel="CCHANNEL1", text=expected_message
        )

    def test_handle_command_status_list_all_no_tasks(self):
        self.mock_task_manager.list_tasks.return_value = {}
        self.handler.handle_command("status", "UUSER1", "CCHANNEL1")
        self.mock_task_manager.list_tasks.assert_called_once()
        self.mock_web_client_instance.chat_postMessage.assert_called_once_with(
            channel="CCHANNEL1", text="No active tasks found."
        )

    def test_handle_command_unknown(self):
        self.handler.handle_command("foobar", "UUSER1", "CCHANNEL1")
        self.mock_web_client_instance.chat_postMessage.assert_called_once_with(
            channel="CCHANNEL1", text="Unknown command: 'foobar'. Type `!crypto help` for available commands."
        )

    # --- Graph and PnL Chart Command Tests ---
    # These require more patching due to file operations and plotting library calls.
    @patch(f'{SLACK_INTERFACE_MODULE_PATH}.os.remove')
    @patch(f'{SLACK_INTERFACE_MODULE_PATH}.os.path.exists')
    @patch(f'{SLACK_INTERFACE_MODULE_PATH}.tempfile.NamedTemporaryFile')
    @patch(f'{SLACK_INTERFACE_MODULE_PATH}.generate_portfolio_graph')
    def test_handle_command_graph_success(self, mock_generate_graph, mock_NamedTemporaryFile,
                                          mock_os_exists, mock_os_remove):
        # Mock file operations
        mock_temp_file_obj = Mock()
        mock_temp_file_obj.name = "dummy_temp_graph.png"
        mock_named_temp_file_cm = MagicMock() # Use MagicMock for context manager __enter__/__exit__
        mock_named_temp_file_cm.__enter__.return_value = mock_temp_file_obj
        mock_NamedTemporaryFile.return_value = mock_named_temp_file_cm

        mock_os_exists.return_value = True # Simulate file exists for cleanup
        mock_generate_graph.return_value = "dummy_temp_graph.png" # Simulate graph generated

        # Simulate task results and portfolio history fetching (using internal _fetch method)
        # We can also mock self.handler._fetch_portfolio_history if we want to control its output directly
        with patch.object(self.handler, '_fetch_portfolio_history', return_value=[(datetime.datetime.now(), 1000.0)]):
            self.handler.handle_command("graph task_g1", "UUSER1", "CCHANNEL1")

        self.mock_task_manager.get_task_results.assert_called_once_with("task_g1")
        mock_NamedTemporaryFile.assert_called_once_with(suffix=".png", delete=False)
        mock_generate_graph.assert_called_once() # Args checked if specific data is passed
        self.mock_web_client_instance.files_upload_v2.assert_called_once_with(
            channel="CCHANNEL1",
            filepath="dummy_temp_graph.png",
            title="Portfolio Value Over Time - Task task_g1",
            initial_comment="Portfolio graph for task `task_g1`:"
        )
        mock_os_exists.assert_called_once_with("dummy_temp_graph.png")
        mock_os_remove.assert_called_once_with("dummy_temp_graph.png")
        self.mock_web_client_instance.chat_postMessage.assert_not_called() # Upload is the message


    @patch(f'{SLACK_INTERFACE_MODULE_PATH}.os.remove')
    @patch(f'{SLACK_INTERFACE_MODULE_PATH}.os.path.exists')
    @patch(f'{SLACK_INTERFACE_MODULE_PATH}.tempfile.NamedTemporaryFile')
    @patch(f'{SLACK_INTERFACE_MODULE_PATH}.generate_pnl_per_trade_graph')
    def test_handle_command_pnl_chart_success(self, mock_generate_pnl_chart, mock_NamedTemporaryFile,
                                              mock_os_exists, mock_os_remove):
        mock_temp_file_obj = Mock()
        mock_temp_file_obj.name = "dummy_temp_pnl.png"
        mock_named_temp_file_cm = MagicMock()
        mock_named_temp_file_cm.__enter__.return_value = mock_temp_file_obj
        mock_NamedTemporaryFile.return_value = mock_named_temp_file_cm

        mock_os_exists.return_value = True
        mock_generate_pnl_chart.return_value = "dummy_temp_pnl.png"

        with patch.object(self.handler, '_fetch_trade_history', return_value=[{'label': 'T1', 'profit': 100}]):
            self.handler.handle_command("pnl_chart task_p1", "UUSER1", "CCHANNEL1")

        self.mock_task_manager.get_task_results.assert_called_once_with("task_p1")
        mock_NamedTemporaryFile.assert_called_once_with(suffix=".png", delete=False)
        mock_generate_pnl_chart.assert_called_once()
        self.mock_web_client_instance.files_upload_v2.assert_called_once_with(
            channel="CCHANNEL1",
            filepath="dummy_temp_pnl.png",
            title="P/L Per Trade - Task task_p1",
            initial_comment="P/L per trade chart for task `task_p1`:"
        )
        mock_os_exists.assert_called_once_with("dummy_temp_pnl.png")
        mock_os_remove.assert_called_once_with("dummy_temp_pnl.png")
        self.mock_web_client_instance.chat_postMessage.assert_not_called()

    def test_handle_command_graph_no_task_id(self):
        self.handler.handle_command("graph", "UUSER1", "CCHANNEL1")
        self.mock_web_client_instance.chat_postMessage.assert_called_once_with(
            channel="CCHANNEL1", text="Error: `task_id` is required. Usage: `!crypto graph <task_id>`"
        )
        self.mock_web_client_instance.files_upload_v2.assert_not_called()

    # Example test for graph generation failure
    @patch(f'{SLACK_INTERFACE_MODULE_PATH}.tempfile.NamedTemporaryFile')
    @patch(f'{SLACK_INTERFACE_MODULE_PATH}.generate_portfolio_graph', return_value=None) # Simulate gen failure
    def test_handle_command_graph_generation_failure(self, mock_generate_graph, mock_NamedTemporaryFile):
        mock_temp_file_obj = Mock()
        mock_temp_file_obj.name = "dummy_temp_graph.png"
        mock_named_temp_file_cm = MagicMock()
        mock_named_temp_file_cm.__enter__.return_value = mock_temp_file_obj
        mock_NamedTemporaryFile.return_value = mock_named_temp_file_cm

        # Mock os.remove and os.path.exists for the finally block if the file is created then deleted
        with patch(f'{SLACK_INTERFACE_MODULE_PATH}.os.remove') as mock_os_remove, \
             patch(f'{SLACK_INTERFACE_MODULE_PATH}.os.path.exists', return_value=True) as mock_os_exists:

            with patch.object(self.handler, '_fetch_portfolio_history', return_value=[(datetime.datetime.now(), 1000.0)]):
                 self.handler.handle_command("graph task_g_fail", "UUSER1", "CCHANNEL1")

        self.mock_web_client_instance.chat_postMessage.assert_called_once_with(
            channel="CCHANNEL1", text="Error: Failed to generate portfolio graph for task `task_g_fail`."
        )
        self.mock_web_client_instance.files_upload_v2.assert_not_called()
        mock_os_remove.assert_called_once() # Temp file should still be cleaned up


    # Test for file upload failure
    @patch(f'{SLACK_INTERFACE_MODULE_PATH}.os.remove')
    @patch(f'{SLACK_INTERFACE_MODULE_PATH}.os.path.exists', return_value=True)
    @patch(f'{SLACK_INTERFACE_MODULE_PATH}.tempfile.NamedTemporaryFile')
    @patch(f'{SLACK_INTERFACE_MODULE_PATH}.generate_portfolio_graph')
    def test_handle_command_graph_upload_failure(self, mock_generate_graph, mock_NamedTemporaryFile,
                                                mock_os_exists, mock_os_remove):
        mock_temp_file_obj = Mock()
        mock_temp_file_obj.name = "dummy_graph_upload_fail.png"
        mock_named_temp_file_cm = MagicMock()
        mock_named_temp_file_cm.__enter__.return_value = mock_temp_file_obj
        mock_NamedTemporaryFile.return_value = mock_named_temp_file_cm

        mock_generate_graph.return_value = "dummy_graph_upload_fail.png"

        # Simulate Slack API error during file upload
        self.mock_web_client_instance.files_upload_v2.return_value = {'ok': False, 'error': 'upload_failed_test'}

        with patch.object(self.handler, '_fetch_portfolio_history', return_value=[(datetime.datetime.now(), 1000.0)]):
            self.handler.handle_command("graph task_g_upload_fail", "UUSER1", "CCHANNEL1")

        self.mock_web_client_instance.files_upload_v2.assert_called_once()
        self.mock_web_client_instance.chat_postMessage.assert_called_once_with(
            channel="CCHANNEL1", text="Error: Failed to upload portfolio graph for task `task_g_upload_fail`."
        )
        mock_os_remove.assert_called_once_with("dummy_graph_upload_fail.png")


if __name__ == '__main__':
    unittest.main()
