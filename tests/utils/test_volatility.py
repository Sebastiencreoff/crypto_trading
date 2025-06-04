import unittest
import numpy as np
from crypto_trading.utils.volatility import calculate_log_return_volatility

class TestVolatilityCalculation(unittest.TestCase):

    def test_enough_data(self):
        prices = [100 + i*0.5 for i in range(21)] # 21 prices
        result = calculate_log_return_volatility(prices, window=20)
        self.assertIsInstance(result, float)
        self.assertTrue(result > 0) # Volatility should generally be positive for non-constant prices

    def test_not_enough_data(self):
        prices_10 = [100.0] * 10
        self.assertIsNone(calculate_log_return_volatility(prices_10, window=20))

        prices_20 = [100.0] * 20
        self.assertIsNone(calculate_log_return_volatility(prices_20, window=20)) # Needs window + 1 prices

        prices_valid_for_window_1 = [100.0, 101.0] # 2 prices
        self.assertIsNotNone(calculate_log_return_volatility(prices_valid_for_window_1, window=1))
        self.assertIsNone(calculate_log_return_volatility(prices_valid_for_window_1, window=2))


    def test_window_zero_or_negative(self):
        prices = [100.0] * 10
        self.assertIsNone(calculate_log_return_volatility(prices, window=0))
        self.assertIsNone(calculate_log_return_volatility(prices, window=-5))

    def test_prices_contain_zero_or_negative(self):
        prices1 = [10.0, 12.0, 0.0, 14.0, 15.0] # Contains zero
        # Check relevant part for log returns: prices[-(window+1):]
        # Window = 3, needs 4 prices. Relevant part for prices1[-(3+1):] i.e. prices1[-4:] = [12.0, 0.0, 14.0, 15.0]
        self.assertIsNone(calculate_log_return_volatility(prices1, window=3))

        prices2 = [10.0, 12.0, -5.0, 14.0, 15.0] # Contains negative
        self.assertIsNone(calculate_log_return_volatility(prices2, window=3))

        prices3 = [0.0, 10.0, 12.0, 14.0, 15.0] # Zero at the start
        # Relevant part for window=3: prices3[-4:] = [10.0, 12.0, 14.0, 15.0] (if series long enough)
        # prices3 is [0,10,12,14,15]. log(10/0) is problem.
        # The check `np.any(relevant_prices_for_log <=0)` should catch this.
        # relevant_prices_for_log = prices[-(window+1):]
        # For window=3, relevant = prices3[-4:] = [0,10,12,14] -> error
        # Actually, prices = [0,10,12,14,15]. window=3. relevant_prices_for_log = prices[-(3+1):] = prices[-4:] = [10,12,14,15]
        # No, relevant_prices_for_log = prices[-(window + 1):] = prices[-(3+1):] = prices[-4:] = [10.0,12.0,14.0,15.0]
        # The issue is in log_returns = np.log(prices[1:] / prices[:-1])
        # prices[1:] = [10,12,14,15], prices[:-1] = [0,10,12,14]
        # prices[1:]/prices[:-1] = [10/0, 12/10, 14/12, 15/14]. Division by zero.
        # The `np.any(prices <=0)` check is before log returns.
        # prices = np.array([0.0, 10.0, 12.0, 14.0, 15.0])
        # np.any(prices <=0) is true.
        # relevant_prices_for_log = prices[-(3+1):] = prices[-4:] = [10,12,14,15] -> this is fine
        # The logic for np.any(relevant_prices_for_log <=0) was to ensure the specific segment used for log calculation is positive.
        # However, the division by zero for prices[1:] / prices[:-1] happens if any price in prices[:-1] is zero.
        # The current check `if np.any(prices <= 0): ... if np.any(relevant_prices_for_log <=0): return None`
        # This needs to be more robust. If any price used in a division `p_i / p_{i-1}` has `p_{i-1} <= 0` or `p_i <= 0`, log will fail.
        # The simplest is: if any price in the whole series segment used for ANY log return is <=0, then fail.
        # The segment of prices used for log returns is prices[-(window+1):]. If any of these are <=0, fail.
        prices3 = [10.0, 12.0, 1.0, 0.0, 15.0] # Zero in a place that makes prices[:-1] contain zero for division
        self.assertIsNone(calculate_log_return_volatility(prices3, window=3)) # window=3, needs 4 prices. uses prices3[-4:] = [12,1,0,15]. log(15/0)

        prices4 = [10.0, 12.0, 5.0, 1.0, 15.0] # All positive
        self.assertIsNotNone(calculate_log_return_volatility(prices4, window=3))


    def test_empty_price_series(self):
        self.assertIsNone(calculate_log_return_volatility([], window=5))

    def test_non_numeric_price_series(self):
        prices_str = [10.0, 'a', 12.0]
        self.assertIsNone(calculate_log_return_volatility(prices_str, window=2))

        prices_obj = [10.0, TestVolatilityCalculation(), 12.0] # type: ignore
        self.assertIsNone(calculate_log_return_volatility(prices_obj, window=2))


    def test_known_values(self):
        prices1 = [100.0] * 5 # e.g. [100,100,100,100,100]
        # window=2. Needs 3 prices. log_returns from prices[-3:] = [100,100,100]
        # log_returns = [log(100/100), log(100/100)] = [0,0]
        # std([0,0]) = 0
        result1 = calculate_log_return_volatility(prices1, window=2)
        self.assertIsNotNone(result1)
        self.assertAlmostEqual(result1, 0.0, places=9)

        prices2 = [100.0, 101.0, 100.0] # window=2. Needs 3 prices. prices[-3:] is the whole list.
        # log_returns = [log(101/100), log(100/101)]
        # log_returns approx [0.00995033085, -0.00995033085]
        # std of this is indeed 0.00995033085
        result2 = calculate_log_return_volatility(prices2, window=2)
        self.assertIsNotNone(result2)
        self.assertAlmostEqual(result2, 0.00995033085, places=7)

        # Test with a slightly longer series and window
        prices3 = [100, 101, 102.01, 103.0301, 104.060401] # Approx 1% increase each time
        # log returns are approx [log(1.01), log(1.01), log(1.01), log(1.01)] which is [0.00995, 0.00995, 0.00995, 0.00995]
        # std of these should be close to 0
        result3 = calculate_log_return_volatility(prices3, window=4)
        self.assertIsNotNone(result3)
        self.assertAlmostEqual(result3, 0.0, places=7) # std of constant values is 0

if __name__ == '__main__':
    unittest.main()
