import unittest
import logging
from unittest.mock import patch
from crypto_trading.algo.bollinger import Bollinger

class TestBollinger(unittest.TestCase):

    def test_bollinger_init_with_dict(self):
        """Test Bollinger initialization with a dictionary."""
        config = {'Bollinger': {'frequency': 25}}
        bollinger = Bollinger(config)
        self.assertEqual(bollinger.frequency, 25)

    def test_bollinger_init_empty_dict(self):
        """Test Bollinger initialization with an empty sub-dictionary."""
        config = {'Bollinger': {}}
        bollinger = Bollinger(config)
        # Accessing private class variable for default, consider making it a public constant if needed for tests
        self.assertEqual(bollinger.frequency, Bollinger._Bollinger__FREQUENCY__)

    def test_bollinger_init_config_key_missing(self):
        """Test Bollinger initialization when 'Bollinger' key is missing."""
        config = {}
        bollinger = Bollinger(config)
        # Should use default frequency if 'Bollinger' key is missing or algo_specific_config is None
        self.assertEqual(bollinger.frequency, Bollinger._Bollinger__FREQUENCY__)

    def test_update_config_new_frequency(self):
        """Test updating Bollinger frequency."""
        config = {'Bollinger': {'frequency': 20}}
        bollinger = Bollinger(config)

        update_params = {'frequency': 30}
        with patch.object(logging, 'info') as mock_log:
            bollinger.update_config(update_params)

        self.assertEqual(bollinger.frequency, 30)
        mock_log.assert_called_once() # Check that logging.info was called

    def test_update_config_empty_dict(self):
        """Test updating Bollinger with an empty dictionary."""
        initial_frequency = 20
        config = {'Bollinger': {'frequency': initial_frequency}}
        bollinger = Bollinger(config)

        bollinger.update_config({}) # Empty update

        self.assertEqual(bollinger.frequency, initial_frequency) # Should remain unchanged

    def test_update_config_key_missing(self):
        """Test updating Bollinger when 'frequency' key is missing in update dict."""
        initial_frequency = 20
        config = {'Bollinger': {'frequency': initial_frequency}}
        bollinger = Bollinger(config)

        update_params = {'other_param': 100} # 'frequency' key is missing
        bollinger.update_config(update_params)

        self.assertEqual(bollinger.frequency, initial_frequency) # Should remain unchanged

if __name__ == '__main__':
    unittest.main()
