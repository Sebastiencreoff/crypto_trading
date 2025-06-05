import unittest
import logging
from unittest.mock import patch
from crypto_trading.algo.average import GuppyMMA

class TestGuppyMMA(unittest.TestCase):

    def test_guppy_init_with_dict(self):
        """Test GuppyMMA initialization with a dictionary."""
        config = {
            'GuppyMMA': {
                'short_term': [3, 5, 7],
                'long_term': [30, 35, 40],
                'buy': 3,
                'sell': 2
            }
        }
        guppy = GuppyMMA(config)
        self.assertTrue(guppy.active)
        self.assertEqual(guppy.short_terms, [3, 5, 7])
        self.assertEqual(guppy.long_terms, [30, 35, 40])
        self.assertEqual(guppy.buy, 3)
        self.assertEqual(guppy.sell, 2)

    def test_guppy_init_empty_dict(self):
        """Test GuppyMMA initialization with an empty sub-dictionary."""
        config = {'GuppyMMA': {}}
        guppy = GuppyMMA(config)
        self.assertTrue(guppy.active)
        self.assertEqual(guppy.short_terms, GuppyMMA.SHORT_TERM_DFT)
        self.assertEqual(guppy.long_terms, GuppyMMA.LONG_TERM_DFT)
        # Default buy/sell is len of default short_terms
        self.assertEqual(guppy.buy, len(GuppyMMA.SHORT_TERM_DFT))
        self.assertEqual(guppy.sell, len(GuppyMMA.SHORT_TERM_DFT))

    def test_guppy_init_config_key_missing(self):
        """Test GuppyMMA initialization when 'GuppyMMA' key is missing."""
        config = {}
        guppy = GuppyMMA(config)
        self.assertFalse(guppy.active) # Should not be active if config is missing
        self.assertIsNone(guppy.short_terms)
        self.assertIsNone(guppy.long_terms)
        self.assertIsNone(guppy.buy)
        self.assertIsNone(guppy.sell)

    def test_update_config_full(self):
        """Test full update of GuppyMMA configuration."""
        config = {'GuppyMMA': {'short_term': [1,2,3], 'long_term': [10,20,30], 'buy': 2, 'sell': 2}}
        guppy = GuppyMMA(config)

        update_params = {
            'short_term': [4, 5, 6, 7],
            'long_term': [40, 50, 60, 70],
            'buy': 4,
            'sell': 3
        }
        with patch.object(logging, 'info') as mock_log:
            guppy.update_config(update_params)

        self.assertEqual(guppy.short_terms, [4, 5, 6, 7])
        self.assertEqual(guppy.long_terms, [40, 50, 60, 70])
        self.assertEqual(guppy.buy, 4)
        self.assertEqual(guppy.sell, 3)
        mock_log.assert_called_once() # Check that logging.info was called

    def test_update_config_partial_short_term(self):
        """Test partial update of GuppyMMA configuration (only short_term)."""
        initial_short = [1,2,3]
        initial_long = [10,20,30]
        initial_buy = 2
        initial_sell = 2
        config = {'GuppyMMA': {'short_term': initial_short, 'long_term': initial_long, 'buy': initial_buy, 'sell': initial_sell}}
        guppy = GuppyMMA(config)

        update_params = {'short_term': [4, 5]}
        guppy.update_config(update_params)

        self.assertEqual(guppy.short_terms, [4, 5])
        self.assertEqual(guppy.long_terms, initial_long) # Should remain unchanged
        self.assertEqual(guppy.buy, initial_buy)         # Should remain unchanged
        self.assertEqual(guppy.sell, initial_sell)       # Should remain unchanged

    def test_update_config_partial_buy_sell(self):
        """Test partial update of GuppyMMA configuration (only buy/sell)."""
        initial_short = [1,2,3]
        initial_long = [10,20,30]
        config = {'GuppyMMA': {'short_term': initial_short, 'long_term': initial_long, 'buy': 2, 'sell': 2}}
        guppy = GuppyMMA(config)

        update_params = {'buy': 3, 'sell': 1}
        guppy.update_config(update_params)

        self.assertEqual(guppy.short_terms, initial_short) # Should remain unchanged
        self.assertEqual(guppy.long_terms, initial_long) # Should remain unchanged
        self.assertEqual(guppy.buy, 3)
        self.assertEqual(guppy.sell, 1)

    def test_update_config_empty_dict(self):
        """Test updating GuppyMMA with an empty dictionary."""
        initial_short = [1,2,3]
        initial_long = [10,20,30]
        initial_buy = 2
        initial_sell = 2
        config = {'GuppyMMA': {'short_term': initial_short, 'long_term': initial_long, 'buy': initial_buy, 'sell': initial_sell}}
        guppy = GuppyMMA(config)

        guppy.update_config({}) # Empty update

        self.assertEqual(guppy.short_terms, initial_short)
        self.assertEqual(guppy.long_terms, initial_long)
        self.assertEqual(guppy.buy, initial_buy)
        self.assertEqual(guppy.sell, initial_sell)

if __name__ == '__main__':
    unittest.main()
