import unittest
import os
import json
from crypto_trading.algo.algoMain import AlgoMain
# We need to import the algorithm classes themselves to check their types or specific attributes
# if AlgoMain stores them directly and we want to inspect them.
from crypto_trading.algo.average import GuppyMMA
from crypto_trading.algo.bollinger import Bollinger
from crypto_trading.algo.moving_average_crossover import MovingAverageCrossover

class TestAlgoMainConfiguration(unittest.TestCase):

    def setUp(self):
        self.config_file_path = "test_algo_main_config.json"
        self.initial_config_data = {
            "GuppyMMA": {"short_term": [3, 5], "long_term": [8, 10], "buy": 1, "sell": 1}, # sell was -1, but Guppy expects positive counts
            "Bollinger": {"frequency": 20},
            "MovingAverageCrossover": {"short_window": 10, "long_window": 30}
        }
        with open(self.config_file_path, 'w') as f:
            json.dump(self.initial_config_data, f)

    def tearDown(self):
        if os.path.exists(self.config_file_path):
            os.remove(self.config_file_path)

    def test_initialization_loads_full_config(self):
        algo_main = AlgoMain(self.config_file_path)

        self.assertEqual(algo_main.full_config_data, self.initial_config_data)
        self.assertEqual(len(algo_main.algo_ifs), 3)

        # Check types and one parameter for each algo to ensure correct instantiation
        self.assertIsInstance(algo_main.algo_ifs[0], GuppyMMA)
        self.assertEqual(algo_main.algo_ifs[0].long_terms, [8, 10])

        self.assertIsInstance(algo_main.algo_ifs[1], Bollinger)
        self.assertEqual(algo_main.algo_ifs[1].frequency, 20)

        self.assertIsInstance(algo_main.algo_ifs[2], MovingAverageCrossover)
        self.assertEqual(algo_main.algo_ifs[2].long_window, 30)

        # max_frequencies depends on the individual algo's max_frequencies()
        # Guppy: max([8,10]) = 10
        # Bollinger: 20
        # MAC: 30
        self.assertEqual(algo_main.max_frequencies, 30)

    def test_update_all_algorithm_configs(self):
        algo_main = AlgoMain(self.config_file_path)

        new_full_config = {
            "GuppyMMA": {"short_term": [4, 6], "long_term": [9, 12], "buy": 2, "sell": 2},
            "Bollinger": {"frequency": 25},
            "MovingAverageCrossover": {"short_window": 15, "long_window": 40}
        }
        algo_main.update_all_algorithm_configs(new_full_config)

        self.assertEqual(algo_main.full_config_data, new_full_config)

        # Check updated params
        self.assertEqual(algo_main.algo_ifs[0].long_terms, [9,12])
        self.assertEqual(algo_main.algo_ifs[0].buy_decision, 2)

        self.assertEqual(algo_main.algo_ifs[1].frequency, 25)

        self.assertEqual(algo_main.algo_ifs[2].long_window, 40)

        # Check recalculated max_frequencies
        # Guppy: 12
        # Bollinger: 25
        # MAC: 40
        self.assertEqual(algo_main.max_frequencies, 40)

    def test_update_all_algorithm_configs_missing_algo_section(self):
        algo_main = AlgoMain(self.config_file_path)

        # Store initial params for comparison
        initial_guppy_long_terms = algo_main.algo_ifs[0].long_terms[:]
        initial_mac_long_window = algo_main.algo_ifs[2].long_window

        partial_new_config = {
            "Bollinger": {"frequency": 22}
            # GuppyMMA and MovingAverageCrossover sections are missing
        }
        algo_main.update_all_algorithm_configs(partial_new_config)

        self.assertEqual(algo_main.full_config_data, partial_new_config) # AlgoMain stores the new config

        # Bollinger should be updated
        self.assertEqual(algo_main.algo_ifs[1].frequency, 22)

        # GuppyMMA and MAC should NOT be updated and retain their original config
        # because their keys were missing in partial_new_config. Their update_config
        # methods were called with an empty dict by AlgoMain in this scenario (as per refactored AlgoMain).
        # This means they should have reset to their *class defaults*, not retained original.

        # Check GuppyMMA (algo_ifs[0]) - should be default values
        self.assertEqual(algo_main.algo_ifs[0].short_terms, GuppyMMA.SHORT_TERM_DFT)
        self.assertEqual(algo_main.algo_ifs[0].long_terms, GuppyMMA.LONG_TERM_DFT)
        self.assertEqual(algo_main.algo_ifs[0].buy_decision, len(GuppyMMA.SHORT_TERM_DFT))


        # Check MovingAverageCrossover (algo_ifs[2]) - should be default values
        self.assertEqual(algo_main.algo_ifs[2].short_window, MovingAverageCrossover.DEFAULT_SHORT_WINDOW)
        self.assertEqual(algo_main.algo_ifs[2].long_window, MovingAverageCrossover.DEFAULT_LONG_WINDOW)

        # Recalculate max_frequencies based on these:
        # Guppy: max(GuppyMMA.LONG_TERM_DFT) (e.g. 60)
        # Bollinger: 22
        # MAC: MovingAverageCrossover.DEFAULT_LONG_WINDOW (e.g. 50)
        expected_max_freq = max(max(GuppyMMA.LONG_TERM_DFT), 22, MovingAverageCrossover.DEFAULT_LONG_WINDOW)
        self.assertEqual(algo_main.max_frequencies, expected_max_freq)

    def test_recalculate_max_frequencies_no_algos(self):
        algo_main = AlgoMain(self.config_file_path)
        algo_main.algo_ifs = [] # Remove all algorithms
        algo_main.recalculate_max_frequencies()
        self.assertEqual(algo_main.max_frequencies, 0)

    def test_recalculate_max_frequencies_one_algo_none_freq(self):
        # Test case where one algo might return None for max_frequencies
        class MockAlgoWithNoneFreq:
            def max_frequencies(self): return None
            def update_config(self, cfg): pass # Dummy

        algo_main = AlgoMain(self.config_file_path) # Normal init
        original_max_freq = algo_main.max_frequencies

        mock_algo = MockAlgoWithNoneFreq()
        algo_main.algo_ifs.append(mock_algo) # Add an algo that returns None for freq

        algo_main.recalculate_max_frequencies()
        # Max frequency should still be determined by the valid algos
        self.assertEqual(algo_main.max_frequencies, original_max_freq)


if __name__ == '__main__':
    unittest.main()
