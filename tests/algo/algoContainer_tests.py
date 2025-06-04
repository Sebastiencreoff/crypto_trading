#!/usr/bin/env python3
import unittest
from unittest.mock import patch, MagicMock, mock_open
import json
import tempfile

from crypto_trading.algo.algoMain import AlgoMain
# Assuming average and bollinger are in crypto_trading.algo
# Forcing the import path for mocks
import crypto_trading.algo.average as average_module
import crypto_trading.algo.bollinger as bollinger_module
import crypto_trading.algo.model as model_module


class TestAlgoMain(unittest.TestCase):

    def create_dummy_config(self, data=None):
        if data is None:
            data = {
                "algo": {
                    "GuppyMMA": {"average": [10, 20]},
                    "Bollinger": {"period": 20}
                },
                "database": {"type": "sqlite"} # Dummy db config
            }

        # Create a temporary file to act as the config file
        # The NamedTemporaryFile must be kept open for json.load to read it
        # It will be closed and deleted when the file object is garbage collected
        # or explicitly closed. For test methods, it's better to manage its lifecycle.
        fp = tempfile.NamedTemporaryFile(mode='w', delete=False)
        json.dump(data, fp)
        fp.close() # Close the file so AlgoMain can open it, especially on Windows
        return fp.name

    @patch(f'{__name__}.model_module.reset') # Mock model.reset
    @patch(f'{__name__}.bollinger_module.Bollinger') # Mock bollinger.Bollinger
    @patch(f'{__name__}.average_module.GuppyMMA') # Mock average.GuppyMMA
    @patch(f'{__name__}.model_module.create') # Mock model.create
    @patch('json.load') # Mock json.load first
    @patch('builtins.open', new_callable=mock_open) # Then mock open
    def test_initialization(self, mock_file_open, mock_json_load, # Order of args matters, must match decorator order from bottom up
                            mock_model_create, mock_guppy_constructor,
                            mock_bollinger_constructor, mock_model_reset_not_used_here):

        dummy_config_data = {
            "algo": {"GuppyMMA": {"average": [3, 5, 7, 10, 12, 15, 30, 35, 40, 45, 50, 60]},
                     "Bollinger": {"period": 20, "nbDev": 2.0}},
            "database": {"type": "sqlite", "path": "test.db"}
        }
        mock_json_load.return_value = dummy_config_data

        # Mock instances returned by constructors
        mock_guppy_instance = MagicMock()
        # algoMain calls x.max_frequencies() which is expected to return an int
        mock_guppy_instance.max_frequencies = MagicMock(return_value=10)
        mock_guppy_constructor.return_value = mock_guppy_instance

        mock_bollinger_instance = MagicMock()
        mock_bollinger_instance.max_frequencies = MagicMock(return_value=20)
        mock_bollinger_constructor.return_value = mock_bollinger_instance

        config_path = "dummy/path/to/config.json" # Path doesn't matter as open is mocked

        algo_main = AlgoMain(config_path)

        # Check that open was called with the config_path by AlgoMain
        mock_file_open.assert_called_once_with(config_path, mode='r')
        # Check that json.load was called with the file object returned by open
        mock_json_load.assert_called_once_with(mock_file_open.return_value)

        mock_guppy_constructor.assert_called_once_with(config_path) # AlgoMain passes the config_path
        mock_bollinger_constructor.assert_called_once_with(config_path) # AlgoMain passes the config_path
        mock_model_create.assert_called_once_with() # model.create() is called with no args

        self.assertEqual(algo_main.max_frequencies, 20) # Max of 10 and 20


    @patch(f'{__name__}.model_module.pricing')
    @patch(f'{__name__}.bollinger_module.Bollinger')
    @patch(f'{__name__}.average_module.GuppyMMA')
    @patch(f'{__name__}.model_module.create')
    @patch('json.load') # Mock json.load first
    @patch('builtins.open', new_callable=mock_open) # Then mock open
    def test_process(self, mock_file_open, mock_json_load, # Order of args matters
                     mock_model_create, mock_guppy_constructor,
                     mock_bollinger_constructor, mock_model_pricing):

        dummy_config_data = {
            "algo": {"GuppyMMA": {}, "Bollinger": {}}, # Keep structure if specific keys are accessed by algos
            "database": {"type": "sqlite"}
        }
        mock_json_load.return_value = dummy_config_data

        mock_guppy_instance = MagicMock()
        mock_guppy_instance.max_frequencies = MagicMock(return_value=10) # Corrected attribute
        mock_guppy_instance.process.return_value = 1 # Guppy says BUY
        mock_guppy_constructor.return_value = mock_guppy_instance

        mock_bollinger_instance = MagicMock()
        mock_bollinger_instance.max_frequencies = MagicMock(return_value=20) # Corrected attribute
        mock_bollinger_instance.process.return_value = 0 # Bollinger says HOLD
        mock_bollinger_constructor.return_value = mock_bollinger_instance

        config_path = "dummy_config.json"
        algo_main = AlgoMain(config_path)

        mock_file_open.assert_called_once_with(config_path, mode='r')
        mock_json_load.assert_called_once_with(mock_file_open.return_value)

        # Mock model.pricing.Pricing instance and its methods
        mock_pricing_instance = MagicMock()
        mock_model_pricing.Pricing.return_value = mock_pricing_instance
        mock_model_pricing.get_last_values.return_value = [90, 95, 100, 105, 110] # Dummy historical values

        current_val = 115
        currency_pair = 'BTC-USD'

        result = algo_main.process(current_value=current_val, currency=currency_pair)

        mock_model_pricing.Pricing.assert_called_once_with(
            currency=currency_pair,
            date_time=unittest.mock.ANY, # datetime.now() is called in production code
            value=current_val
        )
        mock_model_pricing.get_last_values.assert_called_once_with(count=algo_main.max_frequencies, currency=currency_pair)

        mock_guppy_instance.process.assert_called_once_with(current_val, mock_model_pricing.get_last_values.return_value, currency_pair)
        mock_bollinger_instance.process.assert_called_once_with(current_val, mock_model_pricing.get_last_values.return_value, currency_pair)

        self.assertEqual(result, 1) # 1 (Guppy) + 0 (Bollinger) = 1


    @patch(f'{__name__}.model_module.reset')
    @patch(f'{__name__}.bollinger_module.Bollinger')
    @patch(f'{__name__}.average_module.GuppyMMA')
    @patch(f'{__name__}.model_module.create')
    @patch('json.load') # Mock json.load first
    @patch('builtins.open', new_callable=mock_open) # Then mock open
    def test_reset(self, mock_file_open, mock_json_load, # Order of args matters
                   mock_model_create, mock_guppy_constructor,
                   mock_bollinger_constructor, mock_model_reset):

        dummy_config_data = {"algo": {}, "database": {}} # Minimal config
        mock_json_load.return_value = dummy_config_data

        # Setup mock algo instances if their reset methods are also called
        mock_guppy_instance = MagicMock()
        mock_guppy_instance.max_frequencies = MagicMock(return_value=10) # Needs to be callable for init
        mock_guppy_constructor.return_value = mock_guppy_instance

        mock_bollinger_instance = MagicMock()
        mock_bollinger_instance.max_frequencies = MagicMock(return_value=20) # Needs to be callable for init
        mock_bollinger_constructor.return_value = mock_bollinger_instance

        config_path = "dummy_config.json"
        algo_main = AlgoMain(config_path)

        mock_file_open.assert_called_once_with(config_path, mode='r')
        mock_json_load.assert_called_once_with(mock_file_open.return_value)

        algo_main.reset()

        mock_model_reset.assert_called_once()
        # AlgoMain.reset() does not call reset on individual algorithms
        mock_guppy_instance.reset.assert_not_called()
        mock_bollinger_instance.reset.assert_not_called()

if __name__ == '__main__':
    unittest.main()
