import logging
import os
from typing import Optional

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Assuming AppConfig and its nested SlackConfig will be available
from config_management.schemas import SlackConfig

logger = logging.getLogger(__name__)

class SlackNotifier:
    def __init__(self, slack_config: SlackConfig):
        """
        Initializes the SlackNotifier.
        The bot token is sourced from the SLACK_BOT_TOKEN environment variable.
        The default channel ID is sourced from the provided slack_config object.

        Args:
            slack_config: A SlackConfig object containing the default_channel_id.
        """
        self.config = slack_config
        self.client = None
        self.bot_id = None

        bot_token = os.environ.get("SLACK_BOT_TOKEN")
        if not bot_token:
            logger.error("SLACK_BOT_TOKEN environment variable not set. SlackNotifier will be disabled.")
            return

        if not self.config or not self.config.default_channel_id:
            logger.error("SlackConfig with default_channel_id is not properly configured. SlackNotifier will be disabled.")
            return

        try:
            self.client = WebClient(token=bot_token)
            auth_test = self.client.auth_test()
            if not auth_test["ok"]:
                logger.error(f"Slack WebClient auth_test failed: {auth_test.get('error', 'Unknown error')}. SlackNotifier disabled.")
                self.client = None
                self.bot_id = None
                return
            self.bot_id = auth_test.get("bot_id") or auth_test.get("user_id")
            logger.info(f"SlackNotifier initialized successfully. Bot ID: {self.bot_id}, Default Channel: {self.config.default_channel_id}")
        except SlackApiError as e:
            logger.error(f"Slack API error during SlackNotifier initialization: {e.response.get('error', str(e))}. SlackNotifier disabled.", exc_info=True)
            self.client = None
            self.bot_id = None
        except Exception as e:
            logger.error(f"Unexpected error during SlackNotifier initialization: {e}. SlackNotifier disabled.", exc_info=True)
            self.client = None
            self.bot_id = None

    def send_message(self, message: str, channel_id: Optional[str] = None) -> bool:
        """
        Sends a message to a specified Slack channel or the default channel.

        Args:
            message: The message text to send.
            channel_id: The ID of the channel to send the message to.
                        If None, uses the default channel ID.

        Returns:
            True if the message was sent successfully, False otherwise.
        """
        if not self.client:
            logger.warning("Slack client not initialized. Cannot send message.")
            return False

        target_channel_id = channel_id if channel_id else self.config.default_channel_id
        if not target_channel_id:
            logger.error("No target channel ID specified and no default channel ID configured. Cannot send message.")
            return False

        try:
            response = self.client.chat_postMessage(channel=target_channel_id, text=message)
            if response["ok"]:
                logger.info(f"Message successfully sent to Slack channel {target_channel_id}.")
                return True
            else:
                logger.error(f"Error sending Slack message to channel {target_channel_id}: {response.get('error', 'Unknown error')}")
                return False
        except SlackApiError as e:
            logger.error(f"Slack API error while sending message to channel {target_channel_id}: {e.response.get('error', str(e))}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error while sending Slack message to channel {target_channel_id}: {e}", exc_info=True)
            return False

# Example Usage (for testing purposes, not part of the class itself):
if __name__ == '__main__':
    # This example requires a valid bot token and channel ID.
    # Replace with your actual token and channel ID for testing.
    # You would typically load SlackConfig from your main application config.

    class MockSlackConfig(SlackConfig): # Create a mock that fits the expected attributes
        bot_token_for_test: str # Actual token
        default_channel_id_for_test: str # Actual channel ID

        # Override fields from parent SlackConfig for this mock
        webhook_url: str # Will be used as bot_token due to current adaptation
        channel: str # Will be used as default_channel_id

    print("SlackNotifier example usage (requires valid bot token and channel ID):")

    # IMPORTANT: Replace with your actual bot token (starts with xoxb-) and a channel ID.
    test_bot_token = "YOUR_XOXB_BOT_TOKEN"
    test_channel_id = "YOUR_CHANNEL_ID"

    if "YOUR_XOXB_BOT_TOKEN" in test_bot_token or "YOUR_CHANNEL_ID" in test_channel_id:
        print("Please replace YOUR_XOXB_BOT_TOKEN and YOUR_CHANNEL_ID with actual values to test.")
    else:
        # Prepare the mock config based on the temporary adaptation
        # where webhook_url carries the bot token and channel carries the channel_id.
        mock_config = MockSlackConfig(
            bot_token=test_bot_token,
            default_channel_id=test_channel_id,
            # The following are for the MockSlackConfig itself, not used by parent SlackConfig directly
            bot_token_for_test=test_bot_token,
            default_channel_id_for_test=test_channel_id
        )

        logging.basicConfig(level=logging.DEBUG)
        notifier = SlackNotifier(slack_config=mock_config)

        if notifier.client:
            success = notifier.send_message("Hello from SlackNotifier! This is a test message to the default channel.")
            print(f"Message send attempt to default channel successful: {success}")

            # Example: Sending to a specific channel (if different from default and bot is in it)
            # success_specific = notifier.send_message("This is a test to a specific channel.", channel_id="ANOTHER_CHANNEL_ID")
            # print(f"Message send attempt to specific channel successful: {success_specific}")
        else:
            print("SlackNotifier client failed to initialize. Check logs for errors.")
