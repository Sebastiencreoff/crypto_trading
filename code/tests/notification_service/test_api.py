import pytest
from fastapi.testclient import TestClient # Using TestClient for synchronous calls for simplicity
# For async tests with httpx.AsyncClient, refer to FastAPI documentation.
# from httpx import AsyncClient # Example if using AsyncClient directly
from unittest.mock import patch, MagicMock

# Need to adjust path if notification_service is not directly in PYTHONPATH
# For testing, ensure the app can be imported.
# This might require setting PYTHONPATH or using relative imports if structured as a package.
from notification_service.main import app, slack_notifier_instance as global_slack_notifier_instance
from notification_service.slack_client import SlackNotifier
from config_management.schemas import SlackConfig

# It's good practice to ensure a fresh app instance for tests if state can leak,
# but TestClient usually handles this well.
# For more complex scenarios, consider FastAPI's dependency overrides.

@pytest.fixture
def client():
    """Provides a TestClient instance for the notification service app."""
    # If the global_slack_notifier_instance is initialized during app startup via lifespan,
    # we need to ensure it's either mocked before app creation or use dependency_overrides.
    # For this test, we'll assume the global instance can be patched,
    # or the app is set up such that patching works for TestClient.
    with TestClient(app) as c:
        yield c

@pytest.fixture
def mock_slack_notifier():
    """Mocks the SlackNotifier instance used by the API."""
    # Create a mock SlackNotifier
    mock_notifier = MagicMock(spec=SlackNotifier)

    # If the SlackNotifier is initialized in app.lifespan, this direct patch might be tricky.
    # A more robust way for FastAPI is to use dependency overrides.
    # However, for simplicity, if `slack_notifier_instance` is a global variable in main.py,
    # we can try to patch it there.
    # Let's assume `notification_service.main.slack_notifier_instance` is the path to the global var.
    with patch('notification_service.main.slack_notifier_instance', new=mock_notifier) as patched_notifier:
        # Ensure the mocked notifier also has a 'client' attribute if it's checked by endpoints
        # e.g., `if not slack_notifier_instance or not slack_notifier_instance.client:`
        patched_notifier.client = MagicMock() # Simulate an initialized client
        # Also mock the config attribute if used, e.g. `slack_notifier_instance.config.default_channel_id`
        patched_notifier.config = SlackConfig(bot_token="mock_token", default_channel_id="mock_default_C_ID")
        yield patched_notifier


def test_notify_endpoint_success_default_channel(client: TestClient, mock_slack_notifier: MagicMock):
    """Test the /notify endpoint successfully sending to the default channel."""
    mock_slack_notifier.send_message.return_value = True # Simulate successful send

    response = client.post("/notify", json={"message": "Test message to default channel"})

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "success"
    assert json_response["details"] == "Message sent to Slack."
    assert json_response["channel_id"] == mock_slack_notifier.config.default_channel_id

    # Verify that SlackNotifier.send_message was called correctly
    mock_slack_notifier.send_message.assert_called_once_with(
        message="Test message to default channel",
        channel_id=None # Explicitly None to use default
    )

def test_notify_endpoint_success_specific_channel(client: TestClient, mock_slack_notifier: MagicMock):
    """Test the /notify endpoint successfully sending to a specific channel."""
    mock_slack_notifier.send_message.return_value = True
    specific_channel_id = "C123SPECIFIC"

    response = client.post("/notify", json={"message": "Test message to specific channel", "channel_id": specific_channel_id})

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "success"
    assert json_response["channel_id"] == specific_channel_id

    mock_slack_notifier.send_message.assert_called_once_with(
        message="Test message to specific channel",
        channel_id=specific_channel_id
    )

def test_notify_endpoint_slack_send_failure(client: TestClient, mock_slack_notifier: MagicMock):
    """Test the /notify endpoint when SlackNotifier.send_message returns False."""
    mock_slack_notifier.send_message.return_value = False # Simulate failed send

    response = client.post("/notify", json={"message": "Test message that fails to send"})

    assert response.status_code == 500 # Internal Server Error as per current main.py logic
    json_response = response.json()
    assert json_response["detail"] == "Failed to send notification via Slack."

    mock_slack_notifier.send_message.assert_called_once_with(
        message="Test message that fails to send",
        channel_id=None
    )

def test_notify_endpoint_missing_message(client: TestClient):
    """Test the /notify endpoint with missing 'message' field."""
    response = client.post("/notify", json={"channel_id": "C123"}) # Missing 'message'

    assert response.status_code == 422 # Unprocessable Entity for Pydantic validation error
    json_response = response.json()
    assert "detail" in json_response
    assert any(err["loc"] == ["body", "message"] and err["type"] == "missing" for err in json_response["detail"])


def test_notify_endpoint_notifier_not_initialized(client: TestClient, mock_slack_notifier: MagicMock):
    """Test /notify when SlackNotifier client is not initialized (e.g. startup failure)."""
    # Simulate the notifier's client being None (as if init failed)
    mock_slack_notifier.client = None

    response = client.post("/notify", json={"message": "Test message with no notifier client"})

    assert response.status_code == 503 # Service Unavailable
    json_response = response.json()
    assert json_response["detail"] == "Notification service is currently unavailable (Slack client error)."


# To run these tests with pytest, ensure that the FastAPI application's lifespan
# correctly handles the slack_notifier_instance if it's initialized there.
# If `global_slack_notifier_instance` from `notification_service.main` is None before
# the lifespan manager runs (which is typical for TestClient unless app is fully spun up),
# the patch in `mock_slack_notifier` fixture should work as intended.
# If issues arise with patching the global instance due to lifespan complexities,
# FastAPI's dependency_overrides mechanism is the recommended way to inject mocks:
# from notification_service.main import get_slack_notifier_dependency # Assuming you create one
# app.dependency_overrides[get_slack_notifier_dependency] = lambda: mock_notifier_instance
# This requires defining the notifier as a dependency in your path operation function.
# For now, the direct patch of the global variable is simpler if it works for the test setup.
# The current `notification_service.main.py` initializes `slack_notifier_instance` globally
# within the lifespan, so the patch should target this global variable.
# The `client` fixture will trigger the lifespan.

# A note on the `mock_slack_notifier` fixture:
# It patches 'notification_service.main.slack_notifier_instance'.
# The lifespan function in main.py also sets this global variable.
# The order of operations (fixture patching vs. lifespan execution) matters.
# TestClient(app) will run the lifespan. If the patch is applied *before*
# lifespan, lifespan might overwrite it. If *after*, lifespan might have used the real one.
# A robust solution for testing components initialized in lifespan:
# 1. Factor out notifier initialization from lifespan into a dependency function.
# 2. Override this dependency in tests.
# For this iteration, we assume the patching strategy is effective. If tests fail due to
# the real notifier being called, this fixture and test setup for mocking would need refinement
# using FastAPI's dependency_overrides.
# Given the current structure, `slack_notifier_instance` is a global var.
# The patch should ideally be active *during* the request handling,
# overriding what lifespan might have set.
# `patch` in the fixture applies for the duration of the test using the fixture.
# If the app's lifespan creates the real instance *before* the patch can take effect
# for the request, then dependency injection is cleaner.
# Let's assume for now the patch is effective for the test client's requests.
# The current `main.py` sets the global `slack_notifier_instance` in `lifespan`.
# The TestClient will execute this lifespan. The `mock_slack_notifier` fixture
# uses `patch` as a context manager. This patch will be active when the `client.post`
# call is made. So, the `slack_notifier_instance` accessed by the endpoint *should* be the mock.
# This relies on `notification_service.main.slack_notifier_instance` being the exact reference
# the endpoint uses.
# The endpoint code is:
# `if not slack_notifier_instance or not slack_notifier_instance.client:`
# This directly uses the global, so the patch should work.
