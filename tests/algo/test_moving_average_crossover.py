import unittest
import os
import json
from crypto_trading.algo.moving_average_crossover import MovingAverageCrossover

# Helper to create a dummy config file for tests (used by older tests)
def create_dummy_config_file(config_path, short_window=20, long_window=50):
    config_data = {
        "MovingAverageCrossover": {
            "short_window": short_window,
            "long_window": long_window
        }
    }
    with open(config_path, 'w') as f:
        json.dump(config_data, f)

class TestMovingAverageCrossover(unittest.TestCase):

    def setUp(self):
        self.config_file_path = "test_mac_config.json" # Changed name slightly for clarity
        # Ensure there's no old config file from a previous failed run
        if os.path.exists(self.config_file_path):
            os.remove(self.config_file_path)

    def tearDown(self):
        if os.path.exists(self.config_file_path):
            os.remove(self.config_file_path)

    # --- Existing tests, ensure they still work with dict-based init ---
    # Modifying them to pass dicts directly for initialization where appropriate

    def test_initialization_with_dict(self):
        config_section = {"short_window": 10, "long_window": 30}
        algo = MovingAverageCrossover(config_section)
        self.assertEqual(algo.short_window, 10)
        self.assertEqual(algo.long_window, 30)
        self.assertEqual(algo.max_frequencies(), 30)

    def test_initialization_with_file(self): # Keep one test for file-based init
        create_dummy_config_file(self.config_file_path, short_window=10, long_window=30)
        algo = MovingAverageCrossover(self.config_file_path)
        self.assertEqual(algo.short_window, 10)
        self.assertEqual(algo.long_window, 30)

    def test_initialization_invalid_windows_dict(self):
        config_section = {"short_window": 50, "long_window": 20}
        algo = MovingAverageCrossover(config_section)
        self.assertEqual(algo.short_window, MovingAverageCrossover.DEFAULT_SHORT_WINDOW)
        self.assertEqual(algo.long_window, MovingAverageCrossover.DEFAULT_LONG_WINDOW)

    def test_initialization_missing_config_file_path(self): # For path-based init
        algo = MovingAverageCrossover("non_existent_config.json")
        self.assertEqual(algo.short_window, MovingAverageCrossover.DEFAULT_SHORT_WINDOW)
        self.assertEqual(algo.long_window, MovingAverageCrossover.DEFAULT_LONG_WINDOW)
        self.assertEqual(algo.max_frequencies(), MovingAverageCrossover.DEFAULT_LONG_WINDOW)

    def test_initialization_empty_dict(self):
        algo = MovingAverageCrossover({}) # Empty dict
        self.assertEqual(algo.short_window, MovingAverageCrossover.DEFAULT_SHORT_WINDOW)
        self.assertEqual(algo.long_window, MovingAverageCrossover.DEFAULT_LONG_WINDOW)


    # --- New tests for update_config ---
    def test_update_config(self):
        initial_config_section = {"short_window": 10, "long_window": 30}
        algo = MovingAverageCrossover(initial_config_section)

        new_config_section = {"short_window": 15, "long_window": 40}
        algo.update_config(new_config_section)

        self.assertEqual(algo.short_window, 15)
        self.assertEqual(algo.long_window, 40)
        self.assertEqual(algo.max_frequencies(), 40)

    def test_update_config_invalid(self):
        initial_config_section = {"short_window": 10, "long_window": 30}
        algo = MovingAverageCrossover(initial_config_section)

        invalid_config_section = {"short_window": 50, "long_window": 20}
        algo.update_config(invalid_config_section)

        # Should revert to defaults defined in MovingAverageCrossover class
        self.assertEqual(algo.short_window, MovingAverageCrossover.DEFAULT_SHORT_WINDOW)
        self.assertEqual(algo.long_window, MovingAverageCrossover.DEFAULT_LONG_WINDOW)

    def test_update_config_missing_values(self):
        initial_config_section = {"short_window": 10, "long_window": 30}
        algo = MovingAverageCrossover(initial_config_section)

        partial_config_section = {"short_window": 15} # long_window is missing
        algo.update_config(partial_config_section)

        self.assertEqual(algo.short_window, 15)
        # long_window should use its default as defined in update_config/class
        self.assertEqual(algo.long_window, MovingAverageCrossover.DEFAULT_LONG_WINDOW)

    def test_update_config_empty_dict(self):
        initial_config_section = {"short_window": 10, "long_window": 30}
        algo = MovingAverageCrossover(initial_config_section)

        algo.update_config({}) # Empty dict

        # Parameters should revert to defaults
        self.assertEqual(algo.short_window, MovingAverageCrossover.DEFAULT_SHORT_WINDOW)
        self.assertEqual(algo.long_window, MovingAverageCrossover.DEFAULT_LONG_WINDOW)


    # --- Existing process tests, adapted for dict initialization ---
    def test_not_enough_data(self):
        config_section = {"short_window": 5, "long_window": 10}
        algo = MovingAverageCrossover(config_section)
        values = [100.0] * 9
        self.assertEqual(algo.process(100.0, values, "BTC-USD"), 0)

    def test_sma_calculation(self):
        config_section = {"short_window": 2, "long_window": 4}
        algo = MovingAverageCrossover(config_section)
        # _calculate_sma is an internal method, usually tested via `process`
        # but direct tests can be kept if they are valuable.
        data1 = [10.0, 12.0, 14.0, 16.0]
        self.assertEqual(algo._calculate_sma(data1, 4), 13.0)
        data2 = [10.0, 12.0]
        self.assertEqual(algo._calculate_sma(data2, 2), 11.0)
        data3 = [10.0]
        self.assertIsNone(algo._calculate_sma(data3, 2))


    def test_buy_signal(self):
        config_section = {"short_window": 2, "long_window": 3}
        algo = MovingAverageCrossover(config_section)
        values = [10.0, 9.0, 8.0, 12.0]
        self.assertEqual(algo.process(12.0, values, "BTC-USD"), 1)

    def test_sell_signal(self):
        config_section = {"short_window": 2, "long_window": 3}
        algo = MovingAverageCrossover(config_section)
        values = [10.0, 11.0, 12.0, 8.0]
        self.assertEqual(algo.process(8.0, values, "BTC-USD"), -1)

    def test_no_signal_no_crossover(self):
        config_section = {"short_window": 2, "long_window": 3}
        algo = MovingAverageCrossover(config_section)
        values_uptrend_no_cross = [10.0, 12.0, 11.0, 13.0]
        self.assertEqual(algo.process(13.0, values_uptrend_no_cross, "BTC-USD"), 0)
        values_downtrend_no_cross = [12.0, 10.0, 11.0, 9.0]
        self.assertEqual(algo.process(9.0, values_downtrend_no_cross, "BTC-USD"), 0)

    def test_edge_case_data_just_enough_for_long_window(self):
        config_section = {"short_window": 2, "long_window": 3}
        algo = MovingAverageCrossover(config_section)
        values = [10.0, 11.0, 12.0]
        self.assertEqual(algo.process(12.0, values, "BTC-USD"), 0)

    def test_process_with_value_objects(self):
        class MockPricingValue:
            def __init__(self, value):
                self.value = float(value)
        config_section = {"short_window": 2, "long_window": 3}
        algo = MovingAverageCrossover(config_section)
        values_obj = [MockPricingValue(10.0), MockPricingValue(9.0), MockPricingValue(8.0), MockPricingValue(12.0)]
        self.assertEqual(algo.process(12.0, values_obj, "BTC-USD"), 1)
        values_obj_sell = [MockPricingValue(10.0), MockPricingValue(11.0), MockPricingValue(12.0), MockPricingValue(8.0)]
        self.assertEqual(algo.process(8.0, values_obj_sell, "BTC-USD"), -1)

    def test_bad_data_in_values(self):
        config_section = {"short_window": 2, "long_window": 3}
        algo = MovingAverageCrossover(config_section)
        values = [10.0, 9.0, "bad_data", 12.0]
        self.assertEqual(algo.process(12.0, values, "BTC-USD"), 0)

if __name__ == '__main__':
    unittest.main()
