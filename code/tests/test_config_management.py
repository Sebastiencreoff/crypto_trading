import unittest
import os
from pathlib import Path

# Adjust the Python path to include the root directory for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config_management.loader import load_config
from config_management.schemas import AppConfig, DatabaseConfig, ExchangeConfig, SlackConfig, AlgoConfig

class TestConfigManagement(unittest.TestCase):

    def setUp(self):
        # Define the path to the test configuration file
        # Assumes this test script is in 'tests/' and config is in 'config/' relative to project root
        self.test_config_path = Path(__file__).parent.parent / "config" / "central_config.json"

        # Ensure the test config file exists
        if not self.test_config_path.is_file():
            # Create a dummy central_config.json for testing if it doesn't exist
            # This is a fallback, ideally the file from step 4 should be used.
            os.makedirs(self.test_config_path.parent, exist_ok=True)
            with open(self.test_config_path, "w") as f:
                f.write("""
{
    "service_name": "test_service",
    "database": {
        "type": "sqlite",
        "host": "",
        "port": 0,
        "username": "",
        "password": "",
        "name": "./test_db.db"
    },
    "exchanges": [
        {
            "name": "test_exchange",
            "api_key": "dummy_api_key",
            "secret_key": "dummy_secret_key"
        }
    ],
    "slack": {
        "webhook_url": "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX",
        "channel": "#test-channel"
    },
    "algorithms": [
        {
            "name": "test_algo",
            "parameters": {"param1": "value1"}
        }
    ],
    "other_settings": {"setting1": "value1_other"}
}
                """)
            print(f"Warning: Test config file not found, created a dummy one at {self.test_config_path}")


    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        try:
            config = load_config(str(self.test_config_path))
            self.assertIsInstance(config, AppConfig)

            # Verify some basic fields
            self.assertEqual(config.service_name, "crypto_trader_main") # As per central_config.json

            # Database assertions
            self.assertIsInstance(config.database, DatabaseConfig)
            self.assertEqual(config.database.type, "sqlite")
            self.assertTrue(config.database.name.endswith("BINANCE_TRADES.db")) # Check suffix

            # Ensure the base_config_path is set and is a directory
            self.assertIsNotNone(config.base_config_path)
            self.assertTrue(Path(config.base_config_path).is_dir())

            # Check if the database path is resolved correctly (example)
            expected_db_path_abs = (Path(config.base_config_path) / "./BINANCE_TRADES.db").resolve()
            # The loader currently doesn't auto-resolve paths within the model data itself post-load for basic types.
            # It sets `base_config_path`. If `FilePath` Pydantic type is used and it has a validator
            # that uses `base_config_path`, then it would be resolved.
            # For now, we check the raw value.
            self.assertEqual(config.database.name, "./BINANCE_TRADES.db")


            # Exchanges assertions
            self.assertIsInstance(config.exchanges, list)
            self.assertGreater(len(config.exchanges), 0)
            exchange_config = config.exchanges[0]
            self.assertIsInstance(exchange_config, ExchangeConfig)
            self.assertEqual(exchange_config.name, "binance")
            self.assertEqual(exchange_config.api_key, "YOUR_BINANCE_API_KEY") # As per central_config.json

            # Slack assertions (if present in your central_config.json)
            # Updated for new SlackConfig schema: bot_token, default_channel_id
            if config.slack:
                self.assertIsInstance(config.slack, SlackConfig)
                self.assertEqual(config.slack.bot_token, "YOUR_SLACK_XOXB_BOT_TOKEN_PLACEHOLDER")
                self.assertEqual(config.slack.default_channel_id, "YOUR_SLACK_CHANNEL_ID_PLACEHOLDER")

            # Algorithms assertions (if present)
            if config.algorithms:
                self.assertIsInstance(config.algorithms, list)
                algo_config = config.algorithms[0]
                self.assertIsInstance(algo_config, AlgoConfig)
                self.assertEqual(algo_config.name, "default_algo")
                self.assertIn("maxLost", algo_config.parameters)

            # Notification Service URL assertion
            self.assertIsNotNone(config.notification_service_url)
            self.assertEqual(str(config.notification_service_url), "http://notification-service-svc:8001")


        except FileNotFoundError:
            self.fail(f"Test configuration file not found at {self.test_config_path}. "
                      "Ensure 'config/central_config.json' exists.")
        except ValueError as e:
            self.fail(f"Configuration loading failed with error: {e}")

    def test_load_non_existent_config(self):
        """Test loading a non-existent configuration file."""
        with self.assertRaises(FileNotFoundError):
            load_config("non_existent_config.json")

    def test_load_invalid_json_config(self):
        """Test loading a configuration file with invalid JSON."""
        invalid_json_path = Path(__file__).parent / "invalid_config.json"
        with open(invalid_json_path, "w") as f:
            f.write("{'invalid_json': True,}") # Invalid JSON (single quotes, trailing comma)

        with self.assertRaises(ValueError) as context: # Expecting ValueError from json.load or Pydantic
            load_config(str(invalid_json_path))

        self.assertTrue("Error decoding JSON" in str(context.exception) or "validation error" in str(context.exception).lower())

        os.remove(invalid_json_path) # Clean up

    def test_load_config_missing_required_field(self):
        """Test loading a config file missing a required field (e.g., database)."""
        minimal_invalid_path = Path(__file__).parent / "minimal_invalid_config.json"
        with open(minimal_invalid_path, "w") as f:
            f.write("""
{
    "service_name": "missing_db_service"
}
            """)

        with self.assertRaises(ValueError) as context:
            load_config(str(minimal_invalid_path))

        self.assertIn("validation error", str(context.exception).lower())
        self.assertIn("database\n  field required", str(context.exception).lower()) # Pydantic v2 error message

        os.remove(minimal_invalid_path)

if __name__ == '__main__':
    unittest.main()
