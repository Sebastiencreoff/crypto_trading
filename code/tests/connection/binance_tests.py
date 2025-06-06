import unittest
import json
import os
from unittest.mock import patch, mock_open, MagicMock

# Assuming crypto_trading is in PYTHONPATH
from crypto_trading.connection.binance import BinanceConnect
from binance.exceptions import BinanceAPIException, BinanceOrderException

# Define the paths for mock configuration files
MOCK_CONFIG_DIR = "config"
MOCK_API_CONFIG_FILENAME = "testing_binance_api.json"
MOCK_API_CONFIG_PATH = os.path.join(MOCK_CONFIG_DIR, MOCK_API_CONFIG_FILENAME)


class TestBinanceConnect(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.mock_api_config_data = {
            "api_key": "TEST_API_KEY",
            "api_secret": "TEST_API_SECRET"
        }
        # We will mock open for the API config file read within BinanceConnect
        # The main config path passed to BinanceConnect will be MOCK_API_CONFIG_PATH

        # Ensure the mock config directory exists if we were writing a real temp file
        # For mocking 'open', this is not strictly necessary but good practice if not fully mocking 'open'
        if not os.path.exists(MOCK_CONFIG_DIR):
            os.makedirs(MOCK_CONFIG_DIR)

        # This is the config path that BinanceConnect will receive in its __init__
        self.binance_connect_config_path = MOCK_API_CONFIG_PATH

    @patch('binance.client.Client')
    def test_init_success(self, MockBinanceClient):
        """Test successful initialization of BinanceConnect."""
        # Mock 'open' to return our dummy API key configuration
        m = mock_open(read_data=json.dumps(self.mock_api_config_data))
        with patch('builtins.open', m):
            connector = BinanceConnect(config_dict=self.binance_connect_config_path)

            m.assert_called_once_with(self.binance_connect_config_path, 'r')
            MockBinanceClient.assert_called_once_with("TEST_API_KEY", "TEST_API_SECRET")
            self.assertIsNotNone(connector.client)
            self.assertEqual(connector.api_key, "TEST_API_KEY")
            self.assertEqual(connector.api_secret, "TEST_API_SECRET")

    @patch('binance.client.Client')
    def test_init_file_not_found(self, MockBinanceClient):
        """Test initialization when config file is not found."""
        with patch('builtins.open', mock_open()) as m:
            m.side_effect = FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                BinanceConnect(config_dict="non_existent_config.json")

    @patch('binance.client.Client')
    def test_init_key_error(self, MockBinanceClient):
        """Test initialization when config file is missing keys."""
        invalid_api_config_data = {"api_key": "TEST_API_KEY"} # Missing api_secret
        m = mock_open(read_data=json.dumps(invalid_api_config_data))
        with patch('builtins.open', m):
            with self.assertRaises(KeyError):
                BinanceConnect(config_dict=self.binance_connect_config_path)

    @patch('crypto_trading.connection.binance.logger')
    @patch('binance.client.Client')
    def test_get_value_success(self, MockBinanceClientInstance, mock_logger):
        """Test get_value successfully fetches price."""
        mock_client_instance = MockBinanceClientInstance.return_value
        mock_client_instance.get_symbol_ticker.return_value = {'symbol': 'BTCUSDT', 'price': '50000.00'}

        m = mock_open(read_data=json.dumps(self.mock_api_config_data))
        with patch('builtins.open', m):
            connector = BinanceConnect(config_dict=self.binance_connect_config_path)

        price = connector.get_value('BTCUSDT')
        self.assertEqual(price, 50000.0)
        mock_client_instance.get_symbol_ticker.assert_called_once_with(symbol='BTCUSDT')
        mock_logger.info.assert_any_call("Current price of BTCUSDT: 50000.0")

    @patch('crypto_trading.connection.binance.logger')
    @patch('binance.client.Client')
    def test_get_value_api_error(self, MockBinanceClientInstance, mock_logger):
        """Test get_value handles BinanceAPIException."""
        mock_client_instance = MockBinanceClientInstance.return_value
        mock_client_instance.get_symbol_ticker.side_effect = BinanceAPIException("API error")

        m = mock_open(read_data=json.dumps(self.mock_api_config_data))
        with patch('builtins.open', m):
            connector = BinanceConnect(config_dict=self.binance_connect_config_path)

        with self.assertRaises(BinanceAPIException):
            connector.get_value('BTCUSDT')
        mock_logger.error.assert_any_call("Binance API exception while fetching price for BTCUSDT: API error")

    @patch('crypto_trading.connection.binance.logger')
    @patch('binance.client.Client')
    def test_buy_success(self, MockBinanceClientInstance, mock_logger):
        """Test successful market buy order."""
        mock_client_instance = MockBinanceClientInstance.return_value
        mock_order_response = {
            'symbol': 'BTCUSDT', 'orderId': 123, 'status': 'FILLED',
            'executedQty': '0.002', 'cummulativeQuoteQty': '100.0',
            'fills': [{'price': '50000.0', 'qty': '0.002', 'commission': '0.000002', 'commissionAsset': 'BTC'}] # Commission in base asset
        }
        mock_client_instance.order_market_buy.return_value = mock_order_response
        # Mock get_value call within buy
        mock_client_instance.get_symbol_ticker.return_value = {'price': '50000.00'}


        m = mock_open(read_data=json.dumps(self.mock_api_config_data))
        with patch('builtins.open', m):
            connector = BinanceConnect(config_dict=self.binance_connect_config_path)

        quantity_bought, fee = connector.buy(amount=100, currency='BTCUSDT', currency_value=50000.0)

        mock_client_instance.order_market_buy.assert_called_once_with(symbol='BTCUSDT', quoteOrderQty=100)
        self.assertEqual(quantity_bought, 0.002)
        self.assertEqual(fee, 0.000002) # Assuming fee is sum of commissions in base asset
        mock_logger.info.assert_any_call("Bought 0.002 of BTC, Fee: 2e-06")


    @patch('crypto_trading.connection.binance.logger')
    @patch('binance.client.Client')
    def test_buy_order_error(self, MockBinanceClientInstance, mock_logger):
        """Test buy handles BinanceOrderException."""
        mock_client_instance = MockBinanceClientInstance.return_value
        mock_client_instance.order_market_buy.side_effect = BinanceOrderException("Order error")
        # Mock get_value call within buy
        mock_client_instance.get_symbol_ticker.return_value = {'price': '50000.00'}

        m = mock_open(read_data=json.dumps(self.mock_api_config_data))
        with patch('builtins.open', m):
            connector = BinanceConnect(config_dict=self.binance_connect_config_path)

        with self.assertRaises(BinanceOrderException):
            connector.buy(amount=100, currency='BTCUSDT', currency_value=50000.0)
        mock_logger.error.assert_any_call("Binance order exception during buy order for BTCUSDT: Order error")

    @patch('crypto_trading.connection.binance.logger')
    @patch('binance.client.Client')
    def test_sell_success(self, MockBinanceClientInstance, mock_logger):
        """Test successful market sell order."""
        mock_client_instance = MockBinanceClientInstance.return_value
        mock_order_response = {
            'symbol': 'BTCUSDT', 'orderId': 124, 'status': 'FILLED',
            'executedQty': '0.002', 'cummulativeQuoteQty': '100.0',
            'fills': [{'price': '50000.0', 'qty': '0.002', 'commission': '0.1', 'commissionAsset': 'USDT'}] # Commission in quote asset
        }
        mock_client_instance.order_market_sell.return_value = mock_order_response

        m = mock_open(read_data=json.dumps(self.mock_api_config_data))
        with patch('builtins.open', m):
            connector = BinanceConnect(config_dict=self.binance_connect_config_path)

        total_received, fee = connector.sell(amount=0.002, currency='BTCUSDT', currency_value=50000.0)

        mock_client_instance.order_market_sell.assert_called_once_with(symbol='BTCUSDT', quantity=0.002)
        self.assertEqual(total_received, 100.0)
        self.assertEqual(fee, 0.1) # Assuming fee is sum of commissions in quote asset
        mock_logger.info.assert_any_call("Sold 0.002 of BTC. Received: 100.0 USDT, Fee: 0.1")

    @patch('crypto_trading.connection.binance.logger')
    @patch('binance.client.Client')
    def test_sell_order_error(self, MockBinanceClientInstance, mock_logger):
        """Test sell handles BinanceOrderException."""
        mock_client_instance = MockBinanceClientInstance.return_value
        mock_client_instance.order_market_sell.side_effect = BinanceOrderException("Order error")

        m = mock_open(read_data=json.dumps(self.mock_api_config_data))
        with patch('builtins.open', m):
            connector = BinanceConnect(config_dict=self.binance_connect_config_path)

        with self.assertRaises(BinanceOrderException):
            connector.sell(amount=0.002, currency='BTCUSDT', currency_value=50000.0)
        mock_logger.error.assert_any_call("Binance order exception during sell order for BTCUSDT: Order error")

if __name__ == '__main__':
    unittest.main()
