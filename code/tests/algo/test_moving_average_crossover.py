import unittest
import logging
from unittest.mock import patch
from crypto_trading.algo.moving_average_crossover import MovingAverageCrossover

class TestMovingAverageCrossover(unittest.TestCase):

    def test_initialization_with_dict(self):
        """Test MAC initialization with a dictionary."""
        config = {"MovingAverageCrossover": {"short_window": 10, "long_window": 30}}
        algo = MovingAverageCrossover(config)
        self.assertEqual(algo.short_window, 10)
        self.assertEqual(algo.long_window, 30)
        self.assertEqual(algo.max_frequencies(), 30)

    def test_initialization_invalid_windows_dict(self):
        """Test MAC initialization with invalid windows from dict, should use defaults."""
        config = {"MovingAverageCrossover": {"short_window": 50, "long_window": 20}}
        with patch.object(logging, 'error') as mock_log_error:
            algo = MovingAverageCrossover(config)
        self.assertEqual(algo.short_window, 20) # Default
        self.assertEqual(algo.long_window, 50) # Default
        mock_log_error.assert_called_once()

    def test_initialization_empty_sub_dict(self):
        """Test MAC initialization with an empty sub-dictionary, should use defaults."""
        config = {"MovingAverageCrossover": {}}
        algo = MovingAverageCrossover(config)
        self.assertEqual(algo.short_window, 20) # Default
        self.assertEqual(algo.long_window, 50) # Default

    def test_initialization_config_key_missing(self):
        """Test MAC initialization when 'MovingAverageCrossover' key is missing."""
        config = {} # Key missing
        algo = MovingAverageCrossover(config)
        self.assertEqual(algo.short_window, 20) # Default
        self.assertEqual(algo.long_window, 50) # Default

    def test_update_config_valid(self):
        """Test update_config with valid new window sizes."""
        config = {"MovingAverageCrossover": {"short_window": 10, "long_window": 30}}
        algo = MovingAverageCrossover(config)

        update_params = {"short_window": 15, "long_window": 35}
        with patch.object(logging, 'info') as mock_log_info:
            algo.update_config(update_params)

        self.assertEqual(algo.short_window, 15)
        self.assertEqual(algo.long_window, 35)
        mock_log_info.assert_called_once()

    def test_update_config_invalid_windows(self):
        """Test update_config with invalid new window sizes (short >= long)."""
        config = {"MovingAverageCrossover": {"short_window": 10, "long_window": 30}}
        algo = MovingAverageCrossover(config)

        update_params = {"short_window": 40, "long_window": 35} # Invalid
        with patch.object(logging, 'warning') as mock_log_warning:
            algo.update_config(update_params)

        self.assertEqual(algo.short_window, 10) # Should remain unchanged
        self.assertEqual(algo.long_window, 30) # Should remain unchanged
        mock_log_warning.assert_called_once()

    def test_update_config_partial_short_window(self):
        """Test partial update: only short_window, valid."""
        config = {"MovingAverageCrossover": {"short_window": 10, "long_window": 30}}
        algo = MovingAverageCrossover(config)

        update_params = {"short_window": 12}
        algo.update_config(update_params)
        self.assertEqual(algo.short_window, 12)
        self.assertEqual(algo.long_window, 30) # Remains old long

    def test_update_config_partial_long_window(self):
        """Test partial update: only long_window, valid."""
        config = {"MovingAverageCrossover": {"short_window": 10, "long_window": 30}}
        algo = MovingAverageCrossover(config)

        update_params = {"long_window": 32}
        algo.update_config(update_params)
        self.assertEqual(algo.short_window, 10) # Remains old short
        self.assertEqual(algo.long_window, 32)

    def test_update_config_partial_invalid(self):
        """Test partial update: only short_window, but makes it invalid."""
        config = {"MovingAverageCrossover": {"short_window": 10, "long_window": 30}}
        algo = MovingAverageCrossover(config)

        update_params = {"short_window": 35} # New short (35) >= existing long (30)
        with patch.object(logging, 'warning') as mock_log_warning:
            algo.update_config(update_params)
        self.assertEqual(algo.short_window, 10) # Should remain unchanged
        self.assertEqual(algo.long_window, 30) # Should remain unchanged
        mock_log_warning.assert_called_once()

    def test_update_config_empty_dict(self):
        """Test update_config with an empty dictionary."""
        config = {"MovingAverageCrossover": {"short_window": 10, "long_window": 30}}
        algo = MovingAverageCrossover(config)

        algo.update_config({})
        self.assertEqual(algo.short_window, 10) # Should remain unchanged
        self.assertEqual(algo.long_window, 30) # Should remain unchanged

    # --- Existing process tests adapted for dictionary-based initialization ---
    def get_default_config(self, short_window=20, long_window=50):
        return {"MovingAverageCrossover": {"short_window": short_window, "long_window": long_window}}

    def test_not_enough_data(self):
        algo = MovingAverageCrossover(self.get_default_config(short_window=5, long_window=10))
        values = [100.0] * 9 # Data less than long_window
        self.assertEqual(algo.process(100.0, values, "BTC-USD"), 0)

    def test_sma_calculation(self):
        algo = MovingAverageCrossover(self.get_default_config(short_window=2, long_window=4))
        data1 = [10, 12, 14, 16] # Avg = 13
        self.assertEqual(algo._calculate_sma(data1, 4), 13)
        data2 = [10, 12] # Avg = 11
        self.assertEqual(algo._calculate_sma(data2, 2), 11)
        data3 = [10] # Less than window
        self.assertIsNone(algo._calculate_sma(data3, 2))

    def test_buy_signal(self):
        algo = MovingAverageCrossover(self.get_default_config(short_window=2, long_window=3))
        values = [10.0, 9.0, 8.0, 12.0] # 12 is the latest value
        self.assertEqual(algo.process(12.0, values, "BTC-USD"), 1)

    def test_sell_signal(self):
        algo = MovingAverageCrossover(self.get_default_config(short_window=2, long_window=3))
        values = [10.0, 11.0, 12.0, 8.0] # 8 is the latest value
        self.assertEqual(algo.process(8.0, values, "BTC-USD"), -1)

    def test_no_signal_no_crossover(self):
        algo = MovingAverageCrossover(self.get_default_config(short_window=2, long_window=3))
        values_uptrend_no_cross = [10.0, 12.0, 11.0, 13.0]
        self.assertEqual(algo.process(13.0, values_uptrend_no_cross, "BTC-USD"), 0)
        values_downtrend_no_cross = [12.0, 10.0, 11.0, 9.0]
        self.assertEqual(algo.process(9.0, values_downtrend_no_cross, "BTC-USD"), 0)

    def test_edge_case_data_just_enough_for_long_window(self):
        algo = MovingAverageCrossover(self.get_default_config(short_window=2, long_window=3))
        values = [10.0, 11.0, 12.0]
        self.assertEqual(algo.process(12.0, values, "BTC-USD"), 0)

    def test_process_with_value_objects(self):
        class MockPricingValue:
            def __init__(self, value):
                self.value = float(value)
        algo = MovingAverageCrossover(self.get_default_config(short_window=2, long_window=3))
        values_obj = [MockPricingValue(10.0), MockPricingValue(9.0), MockPricingValue(8.0), MockPricingValue(12.0)]
        self.assertEqual(algo.process(12.0, values_obj, "BTC-USD"), 1)
        values_obj_sell = [MockPricingValue(10.0), MockPricingValue(11.0), MockPricingValue(12.0), MockPricingValue(8.0)]
        self.assertEqual(algo.process(8.0, values_obj_sell, "BTC-USD"), -1)

    def test_bad_data_in_values(self):
        algo = MovingAverageCrossover(self.get_default_config(short_window=2, long_window=3))
        values = [10.0, 9.0, "bad_data", 12.0]
        with patch.object(logging, 'error') as mock_log_error: # Expect an error log
            self.assertEqual(algo.process(12.0, values, "BTC-USD"), 0)
            mock_log_error.assert_called_once()


if __name__ == '__main__':
    unittest.main()
