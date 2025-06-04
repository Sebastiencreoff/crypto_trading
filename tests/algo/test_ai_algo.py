import unittest
import logging
from crypto_trading.algo.ai_algo import AIAlgo

class TestAIAlgo(unittest.TestCase):

    def setUp(self):
        # Suppress warning logs during tests for cleaner output if desired
        # logging.disable(logging.WARNING)
        self.config_dict = {
            "AIAlgo": {
                "enabled": True,
                "model_path": "models/non_existent_model.pth" # Force PlaceholderNet
            }
            # Minimal config for AIAlgo initialization.
            # AIAlgo's __init__ expects the main config_dict, from which it extracts its own 'AIAlgo' section.
        }
        # AIAlgo's PlaceholderNet will expect 5 price features + 3 indicator features = 8
        self.ai_algo = AIAlgo(self.config_dict)

        self.current_value = 100.0
        # 4 historical values; current_value will be the 5th price feature for the model.
        self.sufficient_values = [96.0, 97.0, 98.0, 99.0]
        self.currency = "BTC"

        self.full_indicator_signals = {
            'GuppyMMA': 1,
            'Bollinger': -1,
            'MovingAverageCrossover': 0
        }
        self.partial_indicator_signals = { # Missing MovingAverageCrossover
            'GuppyMMA': 1,
            'Bollinger': -1
        }
        self.empty_indicator_signals = {}


    def tearDown(self):
        # Re-enable logging if it was disabled
        # logging.disable(logging.NOTSET)
        pass

    def test_process_with_full_indicators_sufficient_prices(self):
        """
        Tests AIAlgo.process with sufficient price data and all expected indicator signals.
        """
        signal = self.ai_algo.process(self.current_value, self.sufficient_values, self.currency, self.full_indicator_signals)
        self.assertIn(signal, [-1, 0, 1], "Signal should be -1, 0, or 1")

    def test_process_with_partial_indicators_sufficient_prices(self):
        """
        Tests AIAlgo.process with sufficient price data but partially missing indicator signals.
        AIAlgo should use default (0) for missing ones.
        """
        signal = self.ai_algo.process(self.current_value, self.sufficient_values, self.currency, self.partial_indicator_signals)
        self.assertIn(signal, [-1, 0, 1], "Signal should be -1, 0, or 1")

    def test_process_with_empty_indicators_sufficient_prices(self):
        """
        Tests AIAlgo.process with sufficient price data but empty indicator signals.
        AIAlgo should use defaults (0) for all.
        """
        signal = self.ai_algo.process(self.current_value, self.sufficient_values, self.currency, self.empty_indicator_signals)
        self.assertIn(signal, [-1, 0, 1], "Signal should be -1, 0, or 1")

    def test_process_insufficient_price_data_with_indicators(self):
        """
        Tests AIAlgo.process with insufficient historical price data, but with full indicator signals.
        The internal padding logic for price_features in AIAlgo should activate.
        The number of combined features should still match the model's expectation.
        """
        insufficient_values = [99.0] # Only 1 historical value
        signal = self.ai_algo.process(self.current_value, insufficient_values, self.currency, self.full_indicator_signals)
        # AIAlgo's current padding logic should still allow it to proceed and produce a signal.
        self.assertIn(signal, [-1, 0, 1], "Signal should be -1, 0, or 1 even with price padding")

    def test_process_empty_price_data_with_indicators(self):
        """
        Tests AIAlgo.process with an empty list of historical values, but with full indicator signals.
        AIAlgo's padding logic for price_features should activate.
        """
        empty_values = []
        signal = self.ai_algo.process(self.current_value, empty_values, self.currency, self.full_indicator_signals)
        self.assertIn(signal, [-1, 0, 1], "Signal should be -1, 0, or 1 even with empty price list (padding)")

    def test_initialization_without_model_path_key_and_process(self):
        """
        Tests AIAlgo initialization when 'model_path' key is missing from config.
        It should still fallback to PlaceholderNet and process data.
        """
        config_no_model_path = {
            "AIAlgo": {
                "enabled": True
                # model_path key is deliberately missing
            }
        }
        algo_no_path = AIAlgo(config_no_model_path)
        signal = algo_no_path.process(self.current_value, self.sufficient_values, self.currency, self.full_indicator_signals)
        self.assertIn(signal, [-1, 0, 1], "Signal should be -1, 0, or 1 even if model_path key is missing")

if __name__ == '__main__':
    unittest.main()
