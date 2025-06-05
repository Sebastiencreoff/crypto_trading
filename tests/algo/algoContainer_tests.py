#!/usr/bin/env python

import unittest
import json
import os
import logging # Added for potential debugging, can be removed if not used
from crypto_trading.algo.algoMain import AlgoMain
from crypto_trading.algo.ai_algo import AIAlgo
# Import base classes to check attributes, assuming they store config directly
from crypto_trading.algo.average import GuppyMMA
from crypto_trading.algo.bollinger import Bollinger
from crypto_trading.algo.moving_average_crossover import MovingAverageCrossover


class TestAlgoMain(unittest.TestCase):
    CONFIG_FILE_PATH = "test_algo_config.json"

    def setUp(self):
        # Clean up any old config file before a test
        if os.path.exists(self.CONFIG_FILE_PATH):
            os.remove(self.CONFIG_FILE_PATH)

    def tearDown(self):
        # Clean up config file after each test
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
            "AIAlgo": {"enabled": False, "model_path": "dummy/path.pth"} # model_path needed by AIAlgo init
        }
        self.write_config(config_data)

        algo_main = AlgoMain(self.CONFIG_FILE_PATH)

        self.assertFalse(any(isinstance(algo, AIAlgo) for algo in algo_main.algo_ifs), "AIAlgo instance found when it should be disabled.")

        guppy = next((a for a in algo_main.algo_ifs if isinstance(a, GuppyMMA)), None)
        bollinger = next((a for a in algo_main.algo_ifs if isinstance(a, Bollinger)), None)
        mac = next((a for a in algo_main.algo_ifs if isinstance(a, MovingAverageCrossover)), None)

        self.assertIsNotNone(guppy, "GuppyMMA instance not found.")
        self.assertIsNotNone(bollinger, "Bollinger instance not found.")
        self.assertIsNotNone(mac, "MovingAverageCrossover instance not found.")

        # These assertions rely on GuppyMMA, Bollinger, and MovingAverageCrossover classes
        # storing their parameters directly as attributes with these names.
        # This is based on the assumption that their __init__ methods would parse the 'GuppyMMA' (etc.)
        # section of the config_dict passed to them and set attributes accordingly.
        # For example, in GuppyMMA's __init__(self, config_dict):
        #   guppy_conf = config_dict.get('GuppyMMA', {})
        #   self.short_term = guppy_conf.get('short_term')
        #   self.long_term = guppy_conf.get('long_term')
        #   ... etc.
        self.assertEqual(guppy.short_terms, [1, 2, 3])
        self.assertEqual(bollinger.frequency, 100)
        self.assertEqual(mac.short_window, 5)

    def test_algo_main_with_ai_enabled_overrides_configs(self):
        # Config with AI enabled, and some (potentially conflicting) values for other algos in the main config
        # These conflicting values should be ignored by GuppyMMA, Bollinger, MAC if AIAlgo provides defaults.
        config_data = {
            "GuppyMMA": {"short_term": [99, 98], "long_term": [100, 200], "buy": 9, "sell": 9},
            "Bollinger": {"frequency": 500},
            "MovingAverageCrossover": {"short_window": 1, "long_window": 2},
            "AIAlgo": {"enabled": True, "model_path": "dummy/ai_model.pth"} # dummy path is fine, AIAlgo will use PlaceholderNet
        }
        self.write_config(config_data)

        algo_main = AlgoMain(self.CONFIG_FILE_PATH)

        ai_instance = next((algo for algo in algo_main.algo_ifs if isinstance(algo, AIAlgo)), None)
        self.assertIsNotNone(ai_instance, "AIAlgo instance not found when it should be enabled.")

        # These are the default configs AIAlgo should provide
        expected_configs_from_ai = ai_instance.get_target_algo_configs()

        guppy = next((a for a in algo_main.algo_ifs if isinstance(a, GuppyMMA)), None)
        bollinger = next((a for a in algo_main.algo_ifs if isinstance(a, Bollinger)), None)
        mac = next((a for a in algo_main.algo_ifs if isinstance(a, MovingAverageCrossover)), None)

        self.assertIsNotNone(guppy, "GuppyMMA instance not found.")
        self.assertIsNotNone(bollinger, "Bollinger instance not found.")
        self.assertIsNotNone(mac, "MovingAverageCrossover instance not found.")

        # Check that the instances of GuppyMMA, Bollinger, MAC have parameters from AIAlgo's defaults
        self.assertEqual(guppy.short_terms, expected_configs_from_ai['GuppyMMA']['short_term'])
        self.assertEqual(guppy.long_terms, expected_configs_from_ai['GuppyMMA']['long_term'])
        self.assertEqual(guppy.buy, expected_configs_from_ai['GuppyMMA']['buy'])
        self.assertEqual(guppy.sell, expected_configs_from_ai['GuppyMMA']['sell'])

        self.assertEqual(bollinger.frequency, expected_configs_from_ai['Bollinger']['frequency'])

        self.assertEqual(mac.short_window, expected_configs_from_ai['MovingAverageCrossover']['short_window'])
        self.assertEqual(mac.long_window, expected_configs_from_ai['MovingAverageCrossover']['long_window'])

if __name__ == '__main__':
    unittest.main()
