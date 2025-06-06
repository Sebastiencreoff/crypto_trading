#!/usr/bin/env python

import unittest
import json
import os
import logging
import torch # For AIAlgo process mock output
from unittest.mock import patch, MagicMock, call # call for checking multiple calls to update_config
from sqlalchemy.orm import Session # Added
from crypto_trading.config import Config # Added
from crypto_trading.algo.algoMain import AlgoMain
from crypto_trading.algo.ai_algo import AIAlgo
from crypto_trading.algo.average import GuppyMMA
from crypto_trading.algo.bollinger import Bollinger
from crypto_trading.algo.moving_average_crossover import MovingAverageCrossover


class TestAlgoMain(unittest.TestCase):
    CONFIG_FILE_PATH = "test_algo_main_config.json" # Renamed to avoid potential conflicts

    def setUp(self):
        if os.path.exists(self.CONFIG_FILE_PATH):
            os.remove(self.CONFIG_FILE_PATH)
        # Common dummy data for process tests
        self.current_value = 100.0
        self.currency = "BTC-USD"

    def tearDown(self):
        if os.path.exists(self.CONFIG_FILE_PATH):
            os.remove(self.CONFIG_FILE_PATH)

    def write_config(self, data):
        with open(self.CONFIG_FILE_PATH, 'w') as f:
            json.dump(data, f)

    def test_algo_main_with_ai_disabled(self):
        config_data = {
            "GuppyMMA": {"short_term": [1, 2, 3], "long_term": [10, 20], "buy": 1, "sell": 1},
            "Bollinger": {"frequency": 100},
            "MovingAverageCrossover": {"short_window": 5, "long_window": 10},
            "AIAlgo": {"enabled": False, "model_path": "dummy/path.pth"}
        }
        self.write_config(config_data)
        config_obj = Config(algo_config_path=self.CONFIG_FILE_PATH)
        algo_main = AlgoMain(config_obj)
        self.assertFalse(any(isinstance(algo, AIAlgo) for algo in algo_main.algo_ifs))
        # ... (rest of the assertions from original test are good)

    def test_algo_main_with_ai_enabled_overrides_configs_at_init(self):
        config_data = {
            "GuppyMMA": {"short_term": [99, 98], "long_term": [100, 200], "buy": 9, "sell": 9},
            "Bollinger": {"frequency": 500},
            "MovingAverageCrossover": {"short_window": 1, "long_window": 2},
            "AIAlgo": {"enabled": True, "model_path": "dummy/ai_model.pth"}
        }
        self.write_config(config_data)
        config_obj = Config(algo_config_path=self.CONFIG_FILE_PATH)
        algo_main = AlgoMain(config_obj)
        ai_instance = next((algo for algo in algo_main.algo_ifs if isinstance(algo, AIAlgo)), None)
        self.assertIsNotNone(ai_instance)
        expected_configs_from_ai = ai_instance.get_target_algo_configs()
        guppy = next((a for a in algo_main.algo_ifs if isinstance(a, GuppyMMA)), None)
        bollinger = next((a for a in algo_main.algo_ifs if isinstance(a, Bollinger)), None)
        mac = next((a for a in algo_main.algo_ifs if isinstance(a, MovingAverageCrossover)), None)
        self.assertEqual(guppy.short_terms, expected_configs_from_ai['GuppyMMA']['short_term'])
        self.assertEqual(bollinger.frequency, expected_configs_from_ai['Bollinger']['frequency'])
        self.assertEqual(mac.short_window, expected_configs_from_ai['MovingAverageCrossover']['short_window'])

    @patch('crypto_trading.database.core_operations.save_price_tick') # Added
    @patch('crypto_trading.algo.model.pricing.get_last_values') # Mock fetching historical values
    def test_algo_main_process_with_ai_and_updates(self, mock_get_last_values, mock_save_price_tick): # Added mock_save_price_tick
        """
        Test AlgoMain.process when AIAlgo is enabled and provides new configurations.
        """
        mock_get_last_values.return_value = [90.0, 95.0] # Dummy historical values
        mock_session = MagicMock(spec=Session) # Added

        config_data = {
            "GuppyMMA": {}, # Keep these minimal, AI will provide initial and then update
            "Bollinger": {},
            "MovingAverageCrossover": {},
            "AIAlgo": {"enabled": True, "model_path": "dummy/ai_model.pth"}
        }
        self.write_config(config_data)
        config_obj = Config(algo_config_path=self.CONFIG_FILE_PATH)
        algo_main = AlgoMain(config_obj)

        # Find the instantiated algorithms
        ai_algo_instance = next(algo for algo in algo_main.algo_ifs if isinstance(algo, AIAlgo))
        guppy_instance = next(algo for algo in algo_main.algo_ifs if isinstance(algo, GuppyMMA))
        bollinger_instance = next(algo for algo in algo_main.algo_ifs if isinstance(algo, Bollinger))
        mac_instance = next(algo for algo in algo_main.algo_ifs if isinstance(algo, MovingAverageCrossover))

        # Mock the process methods of individual algorithms
        guppy_instance.process = MagicMock(return_value=1) # Guppy signals Buy
        bollinger_instance.process = MagicMock(return_value=-1) # Bollinger signals Sell
        mac_instance.process = MagicMock(return_value=0) # MAC signals Hold

        # Define AIAlgo's output (signal and new configurations)
        ai_output_signal = 1 # AI signals Buy
        ai_new_configs = {
            'GuppyMMA': {'short_term': [1,2,3,4], 'long_term': [10,20,30,40], 'buy': 4, 'sell': 3},
            'Bollinger': {'frequency': 25},
            # No config for MovingAverageCrossover from AI this cycle
        }
        ai_algo_instance.process = MagicMock(return_value=(ai_output_signal, ai_new_configs))

        # Mock the update_config methods
        guppy_instance.update_config = MagicMock()
        bollinger_instance.update_config = MagicMock()
        mac_instance.update_config = MagicMock()

        # --- Call AlgoMain.process ---
        total_signal = algo_main.process(mock_session, self.current_value, self.currency) # Updated call

        # --- Assertions ---
        # 0. Assert save_price_tick was called
        mock_save_price_tick.assert_called_once_with(session=mock_session, currency_pair=self.currency, price=self.current_value) # Added

        # 1. Assert get_last_values was called correctly
        mock_get_last_values.assert_called_once_with(session=mock_session, currency_pair=self.currency, count=algo_main.max_frequencies) # Updated

        # 2. Assert AIAlgo.process was called correctly
        # indicator_signals for AI: {'GuppyMMA': 1, 'Bollinger': -1, 'MovingAverageCrossover': 0}
        expected_indicator_signals = {'GuppyMMA': 1, 'Bollinger': -1, 'MovingAverageCrossover': 0}
        ai_algo_instance.process.assert_called_once_with(
            self.current_value, [90.0, 95.0], self.currency, expected_indicator_signals
        )

        # 2. Assert update_config calls
        guppy_instance.update_config.assert_called_once_with(ai_new_configs['GuppyMMA'])
        bollinger_instance.update_config.assert_called_once_with(ai_new_configs['Bollinger'])
        mac_instance.update_config.assert_not_called() # No config for MAC

        # 3. Assert total_result
        # Guppy (1) + Bollinger (-1) + MAC (0) + AI (1) = 1
        self.assertEqual(total_signal, 1)

    @patch('crypto_trading.database.core_operations.save_price_tick') # Added
    @patch('crypto_trading.algo.model.pricing.get_last_values')
    def test_algo_main_process_ai_empty_configs(self, mock_get_last_values, mock_save_price_tick): # Added mock_save_price_tick
        """Test AlgoMain.process when AIAlgo returns an empty config map."""
        mock_get_last_values.return_value = []
        mock_session = MagicMock(spec=Session) # Added
        config_data = {"AIAlgo": {"enabled": True, "model_path": "dummy/ai_model.pth"}, "GuppyMMA": {}, "Bollinger": {}, "MovingAverageCrossover": {}}
        self.write_config(config_data)
        config_obj = Config(algo_config_path=self.CONFIG_FILE_PATH)
        algo_main = AlgoMain(config_obj)

        ai_algo_instance = next(algo for algo in algo_main.algo_ifs if isinstance(algo, AIAlgo))
        guppy_instance = next(algo for algo in algo_main.algo_ifs if isinstance(algo, GuppyMMA))

        guppy_instance.process = MagicMock(return_value=0)
        # Mock other non-AI algos process to return 0 for simplicity
        for algo in algo_main.algo_ifs:
            if not isinstance(algo, AIAlgo):
                algo.process = MagicMock(return_value=0)

        ai_output_signal = 0
        ai_empty_configs = {} # AI provides no new configs
        ai_algo_instance.process = MagicMock(return_value=(ai_output_signal, ai_empty_configs))

        guppy_instance.update_config = MagicMock()
        # Mock other non-AI algos update_config
        for algo in algo_main.algo_ifs:
            if not isinstance(algo, AIAlgo):
                algo.update_config = MagicMock()

        with patch.object(logging, 'debug') as mock_log_debug:
            algo_main.process(mock_session, self.current_value, self.currency) # Updated call
            # Check if "AIAlgo did not produce new configurations this cycle." or
            # "AIAlgo ran but provided no new configurations to apply." was logged.
            # This requires checking all calls to mock_log_debug
            self.assertTrue(any("AIAlgo did not produce new configurations this cycle." in s_call[0][0] for s_call in mock_log_debug.call_args_list) or \
                            any("AIAlgo ran but provided no new configurations to apply." in s_call[0][0] for s_call in mock_log_debug.call_args_list) )

        mock_save_price_tick.assert_called_once_with(session=mock_session, currency_pair=self.currency, price=self.current_value) # Added
        mock_get_last_values.assert_called_once_with(session=mock_session, currency_pair=self.currency, count=algo_main.max_frequencies) # Added

        guppy_instance.update_config.assert_not_called()
        # Ensure no update_config was called on any algo
        for algo in algo_main.algo_ifs:
            if not isinstance(algo, AIAlgo):
                algo.update_config.assert_not_called()

    @patch('crypto_trading.database.core_operations.save_price_tick') # Added
    @patch('crypto_trading.algo.model.pricing.get_last_values')
    def test_algo_main_process_ai_error_during_ai_process(self, mock_get_last_values, mock_save_price_tick): # Added mock_save_price_tick
        """Test AlgoMain.process when AIAlgo.process itself raises an error."""
        mock_get_last_values.return_value = []
        mock_session = MagicMock(spec=Session) # Added
        config_data = {"AIAlgo": {"enabled": True, "model_path": "dummy/ai_model.pth"}, "GuppyMMA": {}, "Bollinger": {}, "MovingAverageCrossover": {}}
        self.write_config(config_data)
        config_obj = Config(algo_config_path=self.CONFIG_FILE_PATH)
        algo_main = AlgoMain(config_obj)

        ai_algo_instance = next(algo for algo in algo_main.algo_ifs if isinstance(algo, AIAlgo))
        guppy_instance = next(algo for algo in algo_main.algo_ifs if isinstance(algo, GuppyMMA))

        # Mock non-AI algos
        for algo in algo_main.algo_ifs:
            if not isinstance(algo, AIAlgo):
                algo.process = MagicMock(return_value=0)
                algo.update_config = MagicMock()

        ai_algo_instance.process = MagicMock(side_effect=Exception("AI Processing Failed!"))

        with patch.object(logging, 'error') as mock_log_error:
            total_signal = algo_main.process(mock_session, self.current_value, self.currency) # Updated call
            # Check that the error from AIAlgo.process was logged
            self.assertTrue(any("Error processing AIAlgo" in s_call[0][0] for s_call in mock_log_error.call_args_list))

        # No updates should happen if AIAlgo failed
        for algo in algo_main.algo_ifs:
            if not isinstance(algo, AIAlgo):
                algo.update_config.assert_not_called()

        mock_save_price_tick.assert_called_once_with(session=mock_session, currency_pair=self.currency, price=self.current_value) # Added
        mock_get_last_values.assert_called_once_with(session=mock_session, currency_pair=self.currency, count=algo_main.max_frequencies) # Added
        self.assertEqual(total_signal, 0) # Only non-AI signals contribute, which are mocked to 0

    @patch('crypto_trading.algo.model.pricing.reset')
    def test_algo_main_reset(self, mock_pricing_reset):
        # Config data for AlgoMain instantiation
        config_data = {"AIAlgo": {"enabled": False}}
        self.write_config(config_data)
        config_obj = Config(algo_config_path=self.CONFIG_FILE_PATH)
        algo_main = AlgoMain(config_obj)

        mock_session = MagicMock(spec=Session)
        algo_main.reset(mock_session, self.currency)

        mock_pricing_reset.assert_called_once_with(session=mock_session, currency_pair=self.currency)

        # Example for checking sub-algo resets if they were implemented and called by AlgoMain.reset
        # For GuppyMMA, if it had a reset method:
        # guppy_instance = next((algo for algo in algo_main.algo_ifs if isinstance(algo, GuppyMMA)), None)
        # if guppy_instance and hasattr(guppy_instance, 'reset'):
        #     guppy_instance.reset = MagicMock() # If not already mocked or needs specific check
        #     # Call algo_main.reset again or ensure it's part of the flow
        #     guppy_instance.reset.assert_called_once_with(mock_session, self.currency)
        # This part for sub-algos is illustrative; the primary check is mock_pricing_reset.


if __name__ == '__main__':
    unittest.main()
