import pytest
from respx import MockRouter

from config_management.schemas import SlackConfig
from notification_service.slack_client import SlackNotifier

# A valid Slack API URL for chat.postMessage
SLACK_API_URL = "https://www.slack.com/api/chat.postMessage"

@pytest.fixture
def mock_slack_config() -> SlackConfig:
    """Provides a mock SlackConfig for testing."""
    return SlackConfig(
        bot_token="xoxb-test-token-12345",
        default_channel_id="C123DEFAULT"
    )

def test_slack_notifier_initialization_success(mock_slack_config: SlackConfig, respx_mock: MockRouter):
    """Test successful initialization of SlackNotifier."""
    respx_mock.post("https://www.slack.com/api/auth.test").respond(
        status_code=200,
        json={"ok": True, "bot_id": "BTESTBOTID", "user_id": "UTESTUSERID"}
    )

    notifier = SlackNotifier(slack_config=mock_slack_config)
    assert notifier.client is not None
    assert notifier.bot_id == "BTESTBOTID"
    assert notifier.config.default_channel_id == "C123DEFAULT"

def test_slack_notifier_initialization_auth_failure(mock_slack_config: SlackConfig, respx_mock: MockRouter):
    """Test SlackNotifier initialization when auth.test fails."""
    respx_mock.post("https://www.slack.com/api/auth.test").respond(
        status_code=200,
        json={"ok": False, "error": "invalid_auth"}
    )

    notifier = SlackNotifier(slack_config=mock_slack_config)
    assert notifier.client is None

def test_slack_notifier_initialization_missing_token():
    """Test SlackNotifier initialization with missing token."""
    invalid_config = SlackConfig(bot_token="", default_channel_id="C123DEFAULT")
    notifier = SlackNotifier(slack_config=invalid_config)
    assert notifier.client is None

def test_slack_notifier_initialization_missing_channel_id():
    """Test SlackNotifier initialization with missing default channel ID."""
    invalid_config = SlackConfig(bot_token="xoxb-test-token", default_channel_id="")
    # Initialization might still proceed for the client if token is valid,
    # but sending to default channel will fail.
    # Let's assume auth.test would be mocked if client init was the focus.
    # For this test, the critical part is that default_channel_id is empty.
    # The current SlackNotifier logs an error and sets client to None if default_channel_id is missing.
    notifier = SlackNotifier(slack_config=invalid_config)
    assert notifier.client is None # Because constructor checks for default_channel_id too

@pytest.mark.parametrize("channel_id_override", [None, "C456SPECIFIC"])
def test_send_message_success(
    mock_slack_config: SlackConfig,
    respx_mock: MockRouter,
    channel_id_override: str | None
):
    """Test successful message sending to default or specified channel."""
    # Mock auth.test for successful initialization
    respx_mock.post("https://www.slack.com/api/auth.test").respond(
        status_code=200,
        json={"ok": True, "bot_id": "BTESTBOTID"}
    )

    notifier = SlackNotifier(slack_config=mock_slack_config)
    assert notifier.client is not None

    target_channel = channel_id_override if channel_id_override else mock_slack_config.default_channel_id

    # Mock chat.postMessage
    mock_route = respx_mock.post(SLACK_API_URL).respond(
        status_code=200,
        json={"ok": True, "ts": "12345.67890"} # Example successful response
    )

    message_text = "Hello, Slack!"
    success = notifier.send_message(message=message_text, channel_id=channel_id_override)

    assert success is True
    assert mock_route.called is True

    # Check that the request body was as expected
    last_request = mock_route.calls.last.request
    assert last_request is not None

    # httpx typically sends json as bytes, so decode for comparison
    content = last_request.content.decode('utf-8')
    import json
    payload = json.loads(content)

    assert payload["channel"] == target_channel
    assert payload["text"] == message_text

def test_send_message_failure_api_error(mock_slack_config: SlackConfig, respx_mock: MockRouter):
    """Test message sending failure due to Slack API error."""
    respx_mock.post("https://www.slack.com/api/auth.test").respond(json={"ok": True, "bot_id": "BTESTBOTID"})
    notifier = SlackNotifier(slack_config=mock_slack_config)
    assert notifier.client is not None

    # Mock chat.postMessage to return an error
    respx_mock.post(SLACK_API_URL).respond(
        status_code=200, # Slack often returns 200 OK but with an error in JSON body
        json={"ok": False, "error": "channel_not_found"}
    )

    success = notifier.send_message(message="Test message")
    assert success is False

def test_send_message_failure_http_error(mock_slack_config: SlackConfig, respx_mock: MockRouter):
    """Test message sending failure due to HTTP error (e.g., 500)."""
    respx_mock.post("https://www.slack.com/api/auth.test").respond(json={"ok": True, "bot_id": "BTESTBOTID"})
    notifier = SlackNotifier(slack_config=mock_slack_config)
    assert notifier.client is not None

    respx_mock.post(SLACK_API_URL).respond(status_code=500)

    success = notifier.send_message(message="Test message")
    assert success is False

def test_send_message_client_not_initialized(mock_slack_config: SlackConfig, respx_mock: MockRouter):
    """Test send_message when Slack client failed to initialize."""
    # Simulate client initialization failure (e.g., auth.test fails)
    respx_mock.post("https://www.slack.com/api/auth.test").respond(json={"ok": False, "error": "invalid_auth"})
    notifier = SlackNotifier(slack_config=mock_slack_config)
    assert notifier.client is None

    # Try to send a message
    success = notifier.send_message(message="Test message")
    assert success is False

def test_send_message_no_target_channel(respx_mock: MockRouter):
    """Test send_message when no target channel can be determined."""
    # Configure with a valid token but an empty default_channel_id
    # The notifier's constructor currently makes client None if default_channel_id is empty.
    # So this test is similar to client_not_initialized.
    # To test the specific check for target_channel_id in send_message, we would need to
    # allow client initialization with empty default_channel_id.
    # Current SlackNotifier.__init__ design:
    #   if not self.config.bot_token or not self.config.default_channel_id:
    #       self.client = None
    # This means if default_channel_id is empty, client is None.

    config_no_channel = SlackConfig(bot_token="xoxb-valid-token", default_channel_id="")
    respx_mock.post("https://www.slack.com/api/auth.test").respond(json={"ok": True, "bot_id": "BTESTBOTID"}) # Auth passes

    notifier = SlackNotifier(slack_config=config_no_channel)
    assert notifier.client is None # Due to the check in __init__

    success = notifier.send_message(message="Test message", channel_id=None) # No override, default is empty
    assert success is False

    # If __init__ allowed client with empty default_channel_id, this would test the specific check in send_message:
    # notifier.client = WebClient(token="xoxb-valid-token") # Manually set client for test scenario
    # notifier.config.default_channel_id = "" # Ensure it's empty
    # success = notifier.send_message(message="Test message", channel_id=None)
    # assert success is False
    # This part is commented out as it requires altering class logic for testability or a more complex mock.
    # The current test correctly reflects that client will be None.
