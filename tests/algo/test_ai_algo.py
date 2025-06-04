import unittest
import logging # Import logging to suppress or check logs if necessary
from crypto_trading.algo.ai_algo import AIAlgo

class TestAIAlgo(unittest.TestCase):

    def setUp(self):
        # Suppress warning logs during tests for cleaner output, if desired
        # logging.disable(logging.WARNING)
        pass

    def tearDown(self):
        # Re-enable logging if it was disabled
        # logging.disable(logging.NOTSET)
        pass

    def test_process_with_placeholder_model(self):
        """
        Tests the AIAlgo's process method when it falls back to PlaceholderNet
        due to a non-existent model path.
        """
        config_dict = {
            "AIAlgo": {
                "enabled": True,
                "model_path": "models/non_existent_model.pth"
            },
            # Minimal config for AIAlgo initialization
        }
        algo = AIAlgo(config_dict)

        # Test with sufficient data (PlaceholderNet input_size=5)
        # The AIAlgo.process method constructs its input from values + current_value
        # So, if input_size is 5, values should have 4 elements.
        current_value = 100.0
        values = [95.0, 96.0, 97.0, 98.0] # 4 historical values
        currency = "BTC"

        signal = algo.process(current_value, values, currency)
        self.assertIn(signal, [-1, 0, 1], "Signal should be -1, 0, or 1")

    def test_process_with_insufficient_data(self):
        """
        Tests that AIAlgo returns a neutral signal (0) when there is not enough
        historical data for the model.
        """
        config_dict = {
            "AIAlgo": {
                "enabled": True,
                "model_path": "models/non_existent_model.pth"
            }
        }
        algo = AIAlgo(config_dict)

        # Test with insufficient data (e.g., only 2 historical values for input_size=5)
        current_value = 100.0
        values = [98.0, 99.0] # 2 historical values
        currency = "BTC"

        signal = algo.process(current_value, values, currency)
        self.assertEqual(signal, 0, "Signal should be 0 (neutral) for insufficient data")

    def test_process_with_empty_values(self):
        """
        Tests that AIAlgo returns a neutral signal (0) when the values list is empty.
        """
        config_dict = {
            "AIAlgo": {
                "enabled": True,
                "model_path": "models/non_existent_model.pth"
            }
        }
        algo = AIAlgo(config_dict)
        current_value = 100.0
        values = [] # Empty historical values
        currency = "BTC"

        signal = algo.process(current_value, values, currency)
        self.assertEqual(signal, 0, "Signal should be 0 (neutral) for empty values list")

    def test_initialization_without_model_path_key(self):
        """
        Tests AIAlgo initialization when 'model_path' key is missing from config.
        It should still fallback to PlaceholderNet.
        """
        config_dict = {
            "AIAlgo": {
                "enabled": True
                # model_path key is deliberately missing
            }
        }
        algo = AIAlgo(config_dict)
        current_value = 100.0
        values = [95.0, 96.0, 97.0, 98.0]
        currency = "BTC"
        signal = algo.process(current_value, values, currency)
        self.assertIn(signal, [-1, 0, 1], "Signal should be -1, 0, or 1 even if model_path key is missing")


if __name__ == '__main__':
    # Need to be able to run tests from the command line
    # For this to work correctly when run directly, ensure PYTHONPATH is set up
    # so that `crypto_trading.algo.ai_algo` can be imported.
    # Example: PYTHONPATH=$PYTHONPATH:/path/to/your/project/root python tests/algo/test_ai_algo.py
    unittest.main()
