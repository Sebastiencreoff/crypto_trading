import unittest
import os
import json
from crypto_trading.algo.moving_average_crossover import MovingAverageCrossover

# Helper to create a dummy config file for tests
def create_dummy_config(config_path, short_window=20, long_window=50):
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
        self.config_file_path = "test_algo_config.json"
        # Ensure there's no old config file from a previous failed run
        if os.path.exists(self.config_file_path):
            os.remove(self.config_file_path)

    def tearDown(self):
        if os.path.exists(self.config_file_path):
            os.remove(self.config_file_path)

    def test_initialization(self):
        create_dummy_config(self.config_file_path, short_window=10, long_window=30)
        algo = MovingAverageCrossover(self.config_file_path)
        self.assertEqual(algo.short_window, 10)
        self.assertEqual(algo.long_window, 30)
        self.assertEqual(algo.max_frequencies(), 30)

    def test_initialization_invalid_windows(self):
        # Test case where short_window >= long_window, should revert to defaults
        create_dummy_config(self.config_file_path, short_window=50, long_window=20)
        algo = MovingAverageCrossover(self.config_file_path)
        self.assertEqual(algo.short_window, 20) # Default
        self.assertEqual(algo.long_window, 50) # Default

    def test_initialization_missing_config_file(self):
        # Test with a non-existent config file, should use defaults
        algo = MovingAverageCrossover("non_existent_config.json")
        self.assertEqual(algo.short_window, 20) # Default
        self.assertEqual(algo.long_window, 50) # Default
        self.assertEqual(algo.max_frequencies(), 50)


    def test_not_enough_data(self):
        create_dummy_config(self.config_file_path, short_window=5, long_window=10)
        algo = MovingAverageCrossover(self.config_file_path)
        # Data less than long_window
        values = [100.0] * 9
        self.assertEqual(algo.process(100.0, values, "BTC-USD"), 0)

    def test_sma_calculation(self):
        create_dummy_config(self.config_file_path, short_window=2, long_window=4)
        algo = MovingAverageCrossover(self.config_file_path)
        data1 = [10, 12, 14, 16] # Avg = 13
        self.assertEqual(algo._calculate_sma(data1, 4), 13)
        data2 = [10, 12] # Avg = 11
        self.assertEqual(algo._calculate_sma(data2, 2), 11)
        data3 = [10] # Less than window
        self.assertIsNone(algo._calculate_sma(data3, 2))


    def test_buy_signal(self):
        # Short MA (2) crosses above Long MA (4)
        # Period t-1: Short MA = (10+11)/2 = 10.5, Long MA = (10+11+9+8)/4 = 9.5. Short > Long (Mistake here, needs short < long previously)
        # Let's retry:
        # Previous state (values[:-1]):
        # Short MA (window 2) on [10,8]: (10+8)/2 = 9
        # Long MA (window 4) on [10,8,12,14]: (10+8+12+14)/4 = 11
        # So, sma_short_previous (9) < sma_long_previous (11)
        # Current state (values):
        # Add current_value implicitly to `values`. The `values` param *is* the historical series.
        # `current_value` itself is not used for MA calculation in the provided `process` method structure,
        # rather, `values` should contain the *latest* data.
        # Let `values` be the series up to the *current* point.
        # Short MA on [8,15]: (8+15)/2 = 11.5
        # Long MA on [8,12,14,15]: (8+12+14+15)/4 = 12.25 (Mistake, short still not above long)

        # Let's simplify the data for a clear crossover
        # short_window=2, long_window=3
        create_dummy_config(self.config_file_path, short_window=2, long_window=3)
        algo = MovingAverageCrossover(self.config_file_path)

        # Prices: p1, p2, p3, p4
        # Previous state (using p1, p2, p3):
        # values_prev = [10, 9, 8]
        # sma_short_prev = (9+8)/2 = 8.5
        # sma_long_prev = (10+9+8)/3 = 9
        # Condition: sma_short_prev (8.5) < sma_long_prev (9) - MET

        # Current state (using p2, p3, p4):
        # values_curr = [9, 8, 12] (newest value is 12)
        # sma_short_curr = (8+12)/2 = 10
        # sma_long_curr = (9+8+12)/3 = 9.66...
        # Condition: sma_short_curr (10) > sma_long_curr (9.66) - MET

        # The `values` parameter in `process` includes the most recent value.
        # So, for previous calculations, we use values[:-1]

        # History: [10, 9, 8, 12] where 12 is the latest value
        values = [10.0, 9.0, 8.0, 12.0]
        self.assertEqual(algo.process(12.0, values, "BTC-USD"), 1) # Buy signal

    def test_sell_signal(self):
        # Short MA (2) crosses below Long MA (3)
        create_dummy_config(self.config_file_path, short_window=2, long_window=3)
        algo = MovingAverageCrossover(self.config_file_path)

        # Previous state (using p1, p2, p3):
        # values_prev = [10, 11, 12]
        # sma_short_prev = (11+12)/2 = 11.5
        # sma_long_prev = (10+11+12)/3 = 11
        # Condition: sma_short_prev (11.5) > sma_long_prev (11) - MET

        # Current state (using p2, p3, p4):
        # values_curr = [11, 12, 8] (newest value is 8)
        # sma_short_curr = (12+8)/2 = 10
        # sma_long_curr = (11+12+8)/3 = 10.33...
        # Condition: sma_short_curr (10) < sma_long_curr (10.33) - MET

        values = [10.0, 11.0, 12.0, 8.0] # 8 is the latest value
        self.assertEqual(algo.process(8.0, values, "BTC-USD"), -1) # Sell signal

    def test_no_signal_no_crossover(self):
        create_dummy_config(self.config_file_path, short_window=2, long_window=3)
        algo = MovingAverageCrossover(self.config_file_path)

        # Short MA consistently above Long MA
        # Previous: [10,11,9], Short=(11+9)/2=10, Long=(10+11+9)/3=10. Short == Long (No signal)
        # Current: [11,9,10], Short=(9+10)/2=9.5, Long=(11+9+10)/3=10. Short < Long
        # This is a sell if previous was short > long.
        # Let's ensure previous short > long, current short > long
        # Previous: [10,12,11] -> short_prev=(12+11)/2=11.5, long_prev=(10+12+11)/3=11. (short > long)
        # Current: [12,11,13] -> short_curr=(11+13)/2=12, long_curr=(12+11+13)/3=12. (short == long or short slightly > long if numbers are precise)
        values_uptrend_no_cross = [10.0, 12.0, 11.0, 13.0]
        self.assertEqual(algo.process(13.0, values_uptrend_no_cross, "BTC-USD"), 0)

        # Short MA consistently below Long MA
        # Previous: [12,10,11] -> short_prev=(10+11)/2=10.5, long_prev=(12+10+11)/3=11. (short < long)
        # Current: [10,11,9] -> short_curr=(11+9)/2=10, long_curr=(10+11+9)/3=10. (short == long or short < long)
        values_downtrend_no_cross = [12.0, 10.0, 11.0, 9.0]
        self.assertEqual(algo.process(9.0, values_downtrend_no_cross, "BTC-USD"), 0)

    def test_edge_case_data_just_enough_for_long_window(self):
        create_dummy_config(self.config_file_path, short_window=2, long_window=3)
        algo = MovingAverageCrossover(self.config_file_path)
        # Data exactly equals long_window size. Should produce 0 because previous MAs cannot be calculated.
        values = [10.0, 11.0, 12.0]
        self.assertEqual(algo.process(12.0, values, "BTC-USD"), 0)

    def test_process_with_value_objects(self):
        # Test if `process` can handle list of objects with a 'value' attribute
        # This simulates the structure if `values` are lists of `model.pricing.Pricing` instances
        class MockPricingValue:
            def __init__(self, value):
                self.value = float(value)

        create_dummy_config(self.config_file_path, short_window=2, long_window=3)
        algo = MovingAverageCrossover(self.config_file_path)

        # Same data as test_buy_signal, but wrapped in MockPricingValue
        values_obj = [MockPricingValue(10.0), MockPricingValue(9.0), MockPricingValue(8.0), MockPricingValue(12.0)]
        self.assertEqual(algo.process(12.0, values_obj, "BTC-USD"), 1) # Buy signal

        # Same data as test_sell_signal, wrapped
        values_obj_sell = [MockPricingValue(10.0), MockPricingValue(11.0), MockPricingValue(12.0), MockPricingValue(8.0)]
        self.assertEqual(algo.process(8.0, values_obj_sell, "BTC-USD"), -1) # Sell signal

    def test_bad_data_in_values(self):
        create_dummy_config(self.config_file_path, short_window=2, long_window=3)
        algo = MovingAverageCrossover(self.config_file_path)
        values = [10.0, 9.0, "bad_data", 12.0]
        self.assertEqual(algo.process(12.0, values, "BTC-USD"), 0) # Should not crash, return 0

if __name__ == '__main__':
    unittest.main()
