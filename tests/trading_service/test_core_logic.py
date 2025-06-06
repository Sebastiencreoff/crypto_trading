import pytest
from respx import MockRouter
from unittest.mock import patch, MagicMock

# Adjust path as necessary for your project structure
from trading_service.core import _send_notification_sync # Target the function directly
# If testing methods of Trading class that call this, you'd mock _send_notification_sync instead

# Notification service endpoint (as configured in central_config.json usually)
NOTIFICATION_SVC_URL = "http://notification-service-svc:8001" # Example
NOTIFY_ENDPOINT = f"{NOTIFICATION_SVC_URL}/notify"

def test_send_notification_sync_success(respx_mock: MockRouter):
    """Test successful synchronous notification sending."""
    task_id = "test_task_001"
    message = "Test notification message"

    # Mock the HTTP POST request to the notification service
    mock_route = respx_mock.post(NOTIFY_ENDPOINT).respond(
        status_code=200, # Assuming Notification Service returns 200 on success
        json={"status": "success", "details": "Message sent"} # Example response
    )

    _send_notification_sync(message=message, notification_url=NOTIFICATION_SVC_URL, task_id_for_log=task_id)

    assert mock_route.called is True
    last_request = mock_route.calls.last.request
    assert last_request is not None

    content = last_request.content.decode('utf-8')
    import json
    payload = json.loads(content)
    assert payload["message"] == message

def test_send_notification_sync_failure_http_error(respx_mock: MockRouter, caplog):
    """Test notification sending failure due to HTTP error (e.g., 500 from notification service)."""
    task_id = "test_task_002"
    message = "Test notification for HTTP error"

    respx_mock.post(NOTIFY_ENDPOINT).respond(status_code=503) # Simulate notification service being unavailable

    with patch('trading_service.core.logger_core.error') as mock_logger_error: # Patch logger in core.py
        _send_notification_sync(message=message, notification_url=NOTIFICATION_SVC_URL, task_id_for_log=task_id)

        # Check that an error was logged
        assert mock_logger_error.called is True
        # Example: mock_logger_error.assert_any_call(f"Task {task_id}: Failed to send notification. Status: 503, Response: ")
        # The exact log message can be asserted if needed. For now, just check it was called.

def test_send_notification_sync_failure_request_error(respx_mock: MockRouter, caplog):
    """Test notification sending failure due to httpx.RequestError (e.g., network issue)."""
    task_id = "test_task_003"
    message = "Test notification for request error"

    # Simulate a request error (e.g., DNS resolution failure, connection refused)
    respx_mock.post(NOTIFY_ENDPOINT).mock(side_effect=httpx.ConnectError("Connection refused"))

    with patch('trading_service.core.logger_core.error') as mock_logger_error:
        _send_notification_sync(message=message, notification_url=NOTIFICATION_SVC_URL, task_id_for_log=task_id)

        assert mock_logger_error.called is True
        # mock_logger_error.assert_any_call(f"Task {task_id}: HTTP request error sending notification to {NOTIFY_ENDPOINT}: Connection refused")

def test_send_notification_sync_no_url_configured(caplog):
    """Test behavior when notification URL is not configured."""
    task_id = "test_task_004"
    message = "Test notification with no URL"

    with patch('trading_service.core.logger_core.warning') as mock_logger_warning:
        _send_notification_sync(message=message, notification_url=None, task_id_for_log=task_id)

        # Check that a warning was logged about missing URL
        mock_logger_warning.assert_called_once_with(
            f"Task {task_id}: Notification service URL not configured. Cannot send notification: '{message[:50]}...'"
        )

def test_send_notification_sync_empty_url_configured(caplog):
    """Test behavior when notification URL is configured as an empty string."""
    task_id = "test_task_005"
    message = "Test notification with empty URL"

    with patch('trading_service.core.logger_core.warning') as mock_logger_warning:
        _send_notification_sync(message=message, notification_url="", task_id_for_log=task_id)

        mock_logger_warning.assert_called_once_with(
            f"Task {task_id}: Notification service URL not configured. Cannot send notification: '{message[:50]}...'"
        )

# Further tests for the Trading class itself would be more complex and likely
# fall into integration testing, requiring mocking of:
# - Database sessions and operations
# - Exchange connections (connect.buy, connect.sell, connect.get_value)
# - Algorithm interfaces (algo_if.process, security.sell, security.buy)
# - Threading and queue mechanisms if testing the run() method lifecycle.
#
# Example sketch for testing a part of Trading class if it were refactored for easier unit testing:
#
# class TestTradingLogic:
#     @pytest.fixture
#     def mock_app_config(self):
#         config = MagicMock(spec=AppConfig)
#         config.notification_service_url = "http://mock-notification-svc"
#         # ... other necessary AppConfig attributes
#         return config
#
#     @pytest.fixture
#     def mock_exchange_config(self):
#         return MagicMock(spec=ExchangeConfig)
#
#     @pytest.fixture
#     def mock_algo_config(self):
#         return MagicMock(spec=AlgoConfig)
#
#     @pytest.fixture
#     def mock_session(self):
#         return MagicMock(spec=Session) # from sqlalchemy.orm
#
#     def test_trading_initialization(self, mock_app_config, mock_exchange_config, mock_algo_config, mock_session):
#         # This would test the __init__ method of Trading class
#         # Requires mocking connection.SimulationConnect, connection.BinanceConnect, algo.AlgoMain, algo.Security
#         # if they are called during __init__.
#         with patch('trading_service.core.connection.simulation.SimulationConnect') as MockSimConnect, \
#              patch('trading_service.core.connection.binance.BinanceConnect') as MockBinanceConnect, \
#              patch('trading_service.core.algo.AlgoMain') as MockAlgoMain, \
#              patch('trading_service.core.algo.Security') as MockSecurity:
#
#             mock_exchange_config.name = "simulation" # or "binance"
#             # ... set up other mock configs ...
#
#             task_params = {"currency": "BTCUSDT", "transaction_amount": 100}
#             task_id = "init_test_task"
#             stop_event = MagicMock(spec=threading.Event)
#             results_queue = MagicMock(spec=queue.Queue)
#
#             trading_instance = Trading(
#                 app_config=mock_app_config,
#                 exchange_config=mock_exchange_config,
#                 algo_config=mock_algo_config,
#                 task_params=task_params,
#                 session=mock_session,
#                 task_id=task_id,
#                 stop_event=stop_event,
#                 results_queue=results_queue
#             )
#             assert trading_instance.task_id == task_id
#             # Further assertions on initialization
#             if mock_exchange_config.name == "simulation":
#                 MockSimConnect.assert_called_once()
#             else:
#                 MockBinanceConnect.assert_called_once()
#             MockAlgoMain.assert_called_once()
#             MockSecurity.assert_called_once()

# For now, focusing only on _send_notification_sync as it's a new, testable unit.
