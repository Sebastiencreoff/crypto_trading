import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, ANY
import uuid

from trading_service.main import app # Assuming 'app' is the FastAPI instance
from config_management.schemas import AppConfig, ExchangeConfig, AlgoConfig, DatabaseConfig, SlackConfig
from trading_service.models import CreateTaskRequest

# It's essential that the global app_config in trading_service.main is populated for tests.
# This usually happens in the lifespan manager. TestClient(app) should trigger this.

# --- Mocks for external dependencies ---

@pytest.fixture(autouse=True) # Applied to all tests in this module
def mock_db_session():
    """Mocks the database session used by API endpoints."""
    with patch('trading_service.main.get_db_session_from_engine') as mock_get_session_from_engine:
        mock_session = MagicMock()
        # Mock methods used by endpoints or underlying logic if any directly use the session.
        # For example, if create_trading_task directly used the session (it doesn't currently).
        # mock_session.query.return_value...
        mock_get_session_from_engine.return_value = mock_session
        yield mock_session

@pytest.fixture(autouse=True)
def mock_trading_instance_dependencies():
    """Mocks dependencies of the Trading class constructor."""
    with patch('trading_service.core.connection.simulation.SimulationConnect') as MockSimConnect, \
         patch('trading_service.core.connection.binance.BinanceConnect') as MockBinanceConnect, \
         patch('trading_service.core.algo.AlgoMain') as MockAlgoMain, \
         patch('trading_service.core.algo.Security') as MockSecurity, \
         patch('trading_service.core._send_notification_sync') as MockSendNotification: # Mock notification sending

        # Configure default return values or behaviors if needed for mocks
        MockSimConnect.return_value = MagicMock()
        MockBinanceConnect.return_value = MagicMock()
        MockAlgoMain.return_value = MagicMock()
        MockSecurity.return_value = MagicMock()
        MockSendNotification.return_value = None # It's a fire-and-forget sync call

        yield {
            "sim_connect": MockSimConnect,
            "binance_connect": MockBinanceConnect,
            "algo_main": MockAlgoMain,
            "security": MockSecurity,
            "send_notification": MockSendNotification
        }

@pytest.fixture
def test_app_config() -> AppConfig:
    """Provides a sample AppConfig for testing."""
    return AppConfig(
        service_name="test_trading_service",
        database=DatabaseConfig(type="sqlite", host="", port=0, username="", password="", name="./test_trades.db"),
        exchanges=[
            ExchangeConfig(name="binance_test", api_key="test_api_key", secret_key="test_secret", extra_settings={"simulation": False}),
            ExchangeConfig(name="simulation_test", api_key="", secret_key="", extra_settings={"some_sim_param": "value"})
        ],
        slack=SlackConfig(bot_token="test_slack_token", default_channel_id="C123TEST"),
        algorithms=[
            AlgoConfig(name="default_algo_test", parameters={"param1": "value1", "maxLost": {}, "takeProfit": {}, "AIAlgo": {}}),
            AlgoConfig(name="special_algo_test", parameters={"paramX": "valueY", "maxLost": {}, "takeProfit": {}, "AIAlgo": {}})
        ],
        other_settings={"delay_seconds": 1}, # Use short delay for tests if it were real
        notification_service_url="http://mock-notification-service:8001"
    )

@pytest.fixture(autouse=True)
def mock_global_app_config(test_app_config: AppConfig):
    """Mocks the global app_config loaded by trading_service.main"""
    # This patch targets the global variable `app_config` in `trading_service.main`
    # It also needs to mock `global_app_config` from `crypto_trading.config` which is
    # aliased as `app_config` in `trading_service.main` after `init_global_config`.
    # This is a bit complex due to how config is loaded.
    # The ideal way would be dependency injection for config.
    with patch('trading_service.main.app_config', test_app_config), \
         patch('crypto_trading.config.app_config', test_app_config): # Patch where it's originally set
        # Also, prevent init_global_config from running its full course if it loads real files
        with patch('crypto_trading.config.load_config', return_value=test_app_config):
             with patch('trading_service.main.init_global_config') as mock_init_global_config:
                mock_init_global_config.return_value = test_app_config # Ensure it "provides" the config
                yield

# --- Test Client ---
@pytest.fixture
def client():
    """Provides a TestClient instance for the trading service app."""
    # The TestClient will initialize the app, triggering the lifespan manager.
    # The mock_global_app_config fixture should ensure that the lifespan manager
    # uses our test_app_config.
    with TestClient(app) as c:
        # Clear active_tasks before each test if it's not handled in lifespan/shutdown
        from trading_service.main import active_tasks
        active_tasks.clear()
        yield c
    # Clear active_tasks after each test to ensure isolation
    from trading_service.main import active_tasks
    active_tasks.clear()


# --- API Tests ---

def test_create_trading_task_success(client: TestClient, mock_trading_instance_dependencies, test_app_config: AppConfig):
    """Test successful creation of a trading task."""
    task_payload = CreateTaskRequest(
        currency_pair="BTC/USDT",
        transaction_amount=100.0,
        exchange_name="simulation_test", # Must match one in test_app_config.exchanges
        algo_name="default_algo_test"    # Must match one in test_app_config.algorithms
    )

    # Mock the Trading class's run method to prevent actual execution
    with patch('trading_service.core.Trading.run') as mock_trading_run:
        mock_trading_run.return_value = None

        response = client.post("/tasks", json=task_payload.model_dump())

    assert response.status_code == 201
    json_response = response.json()
    assert "task_id" in json_response
    assert json_response["message"] == "Trading task created."
    assert json_response["details"]["currency_pair"] == task_payload.currency_pair
    assert json_response["details"]["status"] == "starting" # As set in main.py

    # Verify that a task was added to active_tasks (check by task_id)
    from trading_service.main import active_tasks
    assert json_response["task_id"] in active_tasks
    task_data = active_tasks[json_response["task_id"]]
    assert task_data is not None
    assert task_data["info_model"].currency_pair == task_payload.currency_pair

    # Check that Trading constructor was called with expected AppConfig parts
    # This requires inspecting how Trading class is instantiated in main.py
    # For example, check if the correct exchange_config was passed.
    # The actual 'Trading' instance is in task_data['instance']
    # Its __init__ args are harder to get directly without more mocking/spying.
    # However, we can check if the correct connection mock was called if exchange name implies it.
    mock_trading_instance_dependencies["sim_connect"].assert_called_once() # Since "simulation_test" was used
    mock_trading_instance_dependencies["algo_main"].assert_called_once()
    mock_trading_instance_dependencies["security"].assert_called_once()

    # Check if the thread was started
    assert task_data["thread"].is_alive() is True # Thread should be running (mocked run)

    # Ensure the background thread's run method was called (or at least attempted)
    # This is tricky as it's in a separate thread.
    # If Trading.run is patched, its call can be asserted after a short delay or by joining the thread (not ideal in unit test).
    # For now, mock_trading_run.assert_called_once() might not work immediately if thread hasn't run it yet.
    # This test assumes that if thread.start() is called, it's good enough for "creation" success.

def test_create_task_invalid_exchange(client: TestClient, test_app_config: AppConfig):
    """Test task creation with an exchange not in config."""
    task_payload = CreateTaskRequest(
        currency_pair="BTC/USDT",
        transaction_amount=100.0,
        exchange_name="unknown_exchange", # This exchange is not in test_app_config
        algo_name="default_algo_test"
    )
    response = client.post("/tasks", json=task_payload.model_dump())
    assert response.status_code == 400
    assert "Exchange 'unknown_exchange' not configured" in response.json()["detail"]

def test_create_task_invalid_algorithm(client: TestClient, test_app_config: AppConfig):
    """Test task creation with an algorithm not in config."""
    task_payload = CreateTaskRequest(
        currency_pair="BTC/USDT",
        transaction_amount=100.0,
        exchange_name="simulation_test",
        algo_name="unknown_algo" # This algo is not in test_app_config
    )
    response = client.post("/tasks", json=task_payload.model_dump())
    assert response.status_code == 400
    assert "Algorithm 'unknown_algo' not configured" in response.json()["detail"]


@pytest.fixture
def setup_active_task(client: TestClient, mock_trading_instance_dependencies):
    """Helper fixture to create a task and return its ID and mock objects."""
    task_payload = CreateTaskRequest(
        currency_pair="ETH/USDT",
        transaction_amount=50.0,
        exchange_name="simulation_test",
        algo_name="default_algo_test"
    )
    # Mock the run method of the Trading instance that will be created
    # The instance itself is created inside the endpoint. We need to ensure all
    # instances of Trading have their run method mocked for this test.
    with patch('trading_service.core.Trading.run') as mock_run:
        mock_run.return_value = None # Prevent actual execution

        response = client.post("/tasks", json=task_payload.model_dump())
    assert response.status_code == 201
    task_id = response.json()["task_id"]

    from trading_service.main import active_tasks
    assert task_id in active_tasks
    # Return task_id and the specific Trading instance's mock if needed,
    # or the shared mocks from mock_trading_instance_dependencies
    return task_id, active_tasks[task_id]["instance"], active_tasks[task_id] # task_data

def test_get_task_status_found(client: TestClient, setup_active_task):
    """Test getting status for an existing task."""
    task_id, _, _ = setup_active_task

    response = client.get(f"/tasks/{task_id}")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["task_id"] == task_id
    assert json_response["status"] == "starting" # Initial status after creation
    assert json_response["currency_pair"] == "ETH/USDT"

def test_get_task_status_not_found(client: TestClient):
    """Test getting status for a non-existent task."""
    non_existent_task_id = str(uuid.uuid4())
    response = client.get(f"/tasks/{non_existent_task_id}")
    assert response.status_code == 404
    assert response.json()["detail"] == "Task not found."

def test_stop_trading_task_success(client: TestClient, setup_active_task):
    """Test successfully stopping a trading task."""
    task_id, trading_instance_mock, task_data = setup_active_task

    # Mock the stop method of the specific Trading instance
    # trading_instance_mock is the actual instance from active_tasks.
    # We need to ensure its stop method is a MagicMock if we want to assert on it.
    # The instance itself is real, but its dependencies (like _send_notification_sync) are mocked.
    # Let's spy on its `stop` method if it's not already a mock.
    # For simplicity, we assume Trading.stop() works as intended.
    # The stop_event within task_data is the real one.

    # Ensure thread is "alive" for stop to make sense
    task_data['thread'] = MagicMock(spec=threading.Thread)
    task_data['thread'].is_alive.return_value = True


    with patch.object(trading_instance_mock, 'stop', wraps=trading_instance_mock.stop) as mock_instance_stop_method:
        response = client.post(f"/tasks/{task_id}/stop")
        assert response.status_code == 200
        assert response.json()["message"] == "Stop signal sent. Task will process shutdown."

        # Verify that the Trading instance's stop method was called
        mock_instance_stop_method.assert_called_once()
        # Verify that the stop_event was set
        assert task_data["stop_event"].is_set() is True

def test_stop_task_not_found(client: TestClient):
    non_existent_task_id = str(uuid.uuid4())
    response = client.post(f"/tasks/{non_existent_task_id}/stop")
    assert response.status_code == 404

# TODO:
# - Test /tasks/{task_id}/profits endpoint (requires mocking trading_instance.profits)
# - Test /tasks/{task_id}/reset endpoint (requires mocking trading_instance.reset_trading_state and task state)
# - Test status updates from results_queue in get_task_status
# - Test behavior when thread is not alive in get_task_status and stop_trading_task
# - More detailed checks for Trading class constructor calls if possible.
# - Test error handling during task creation more deeply (e.g., if Trading init fails).

# Note on mocking app_config:
# The current approach with `patch('trading_service.main.app_config', test_app_config)`
# and `patch('crypto_trading.config.app_config', test_app_config)` along with
# patching `init_global_config` and `load_config` is an attempt to control the config
# used by the TestClient and the FastAPI app during tests. This can be sensitive
# to the exact structure and import style of the main application.
# If `trading_service.main.app_config` is imported elsewhere as `from trading_service.main import app_config`,
# those modules would get the state of `app_config` at the time of their import.
# Using FastAPI's dependency injection for configuration is a more robust pattern for testability.
# For now, this direct patching is assumed to work for the scope of these tests.
