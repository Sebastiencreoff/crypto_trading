#!/usr/bin/env python3

import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import crypto_trading.connection.coinBase as coinbase
# Import the error for testing, assuming this is a plausible error for unknown currency
from coinbase.wallet import error as coinbase_error # For coinbase.wallet.error.APIError

# Path for patching the Client constructor
CLIENT_PATH = 'coinbase.wallet.client.Client'

class CoinBaseConnectTest(unittest.TestCase):

    def coinBase_config(self, delete=True, simulation=True):
        # Create a temporary file to act as the config file
        fp = tempfile.NamedTemporaryFile(mode='w', delete=delete, suffix=".json")
        # Write dummy api_key/secret as they are expected by CoinBaseConnect's config loading
        # The actual values won't be used for API calls due to mocking.
        json.dump({'api_key': 'mock_api_key',
                   'api_secret': 'mock_api_secret',
                   'simulation': simulation}, fp)
        fp.seek(0) # Rewind to the beginning of the file
        return fp

    def setUp(self):
        # Patch coinbase.wallet.client.Client
        self.client_patcher = patch(CLIENT_PATH)
        self.mock_client_constructor = self.client_patcher.start()
        self.mock_client_instance = self.mock_client_constructor.return_value

        # Configure default successful responses for methods called during CoinBaseConnect initialization

        # Mock for get_payment_methods
        # The data items should be dictionaries if CoinBaseConnect uses .get() on them
        self.raw_pm_data = [{'type': 'fiat_account', 'id': 'mock_fiat_id', 'name': 'Mock EUR Wallet', 'currency': 'EUR', 'primary_buy': True, 'primary_sell': True, 'allow_buy': True, 'allow_sell': True, 'allow_deposit': True, 'allow_withdraw': True}]
        self.mock_client_instance.get_payment_methods.return_value = MagicMock(data=self.raw_pm_data)

        # Mock for get_accounts (default good response)
        # The data items should be dictionaries if CoinBaseConnect uses .get() or direct access on them
        # However, CoinBaseConnect's _get_account_id uses direct attribute access (account['currency'])
        # So, these should be MagicMocks or objects that support attribute access.
        # Let's make them MagicMocks as originally intended.
        # Correction: _get_account_id uses account['currency'], so items should be dicts or dict-like.
        self.raw_account_data = [
            {'id': 'btc_acc_id', 'currency': 'BTC', 'balance': {'amount':'1.0', 'currency':'BTC'}}, # balance can be dict
            {'id': 'eur_acc_id', 'currency': 'EUR', 'balance': {'amount':'1000', 'currency':'EUR'}}
        ]
        # No need to convert to MagicMock instances if direct dict access is used by the code.
        self.mock_client_instance.get_accounts.return_value = MagicMock(data=self.raw_account_data)

    def tearDown(self):
        self.client_patcher.stop()

    # Test methods will be refactored/added below
    # Original test_unknownCurrency will be refactored
    # Original test_execute will be refactored
    # Original test_error will be removed or re-purposed

    # New test based on previous finding about NameError for fiat account
    def test_init_fails_no_fiat_account(self):
        # Override get_payment_methods mock for this specific test
        self.mock_client_instance.get_payment_methods.return_value = MagicMock(data=[]) # No payment methods

        with self.coinBase_config() as fp:
            with self.assertRaisesRegex(NameError, "Fiat account not found or failed to retrieve payment methods."):
                coinbase.CoinBaseConnect(fp.name)

    def test_init_fails_only_non_fiat_account(self):
        raw_pm_data_non_fiat = [{'type': 'crypto_account', 'id': 'mock_crypto_id', 'name': 'Mock BTC Wallet', 'currency': 'BTC'}]
        mock_payment_methods_non_fiat = [MagicMock(**pm_data) for pm_data in raw_pm_data_non_fiat]
        self.mock_client_instance.get_payment_methods.return_value = MagicMock(data=mock_payment_methods_non_fiat)

        with self.coinBase_config() as fp:
            with self.assertRaisesRegex(NameError, "Fiat account not found or failed to retrieve payment methods."):
                coinbase.CoinBaseConnect(fp.name)

    def test_get_value_unknown_currency_api_error(self):
        # Test how get_value handles an APIError from the client library for an unknown currency
        mock_response = MagicMock()
        mock_response.status_code = 404 # Example status code
        # The coinbase library's build_api_error tries to access response.json()['errors'][0]
        mock_response.json.return_value = {'errors': [{'id': 'not_found', 'message': 'Unknown currency'}]}

        # Instantiate the error correctly for side_effect
        # APIError(response, id, message, json_body=None) - response is the requests.Response object
        # The id and message are typically derived from the response body by the library itself.
        # For testing, we can simulate the error object that would be built.
        # Or, more simply, if APIError can be raised with just a message for testing:
        # For now, let's assume the error constructor needs what build_api_error would pass.
        # build_api_error uses blob.get('id'), blob.get('message') where blob is response.json() or response.json().get('errors')[0]
        # So, coinbase_error.APIError(response, id='not_found', message='Unknown currency') might be closer.
        # The actual constructor is APIError(response, message=None, id=None, json_body=None)
        # Let's use a simpler error for the side_effect if possible, or a more specific one like NotFoundError
        # For example: coinbase_error.NotFoundError(response, message="Not found")
        # Checking coinbase.wallet.error.py, NotFoundError is a subclass of APIError.
        # NotFoundError(response, message=None, id=None, json_body=None)
        # Let's use NotFoundError for more specificity if it fits the "Unknown currency" case from API.
        # If get_exchange_rates for an unknown currency gives 404, NotFoundError is appropriate.
        self.mock_client_instance.get_exchange_rates.side_effect = coinbase_error.NotFoundError(
            response=mock_response, # Pass the mock response
            message="Mocked Not Found: Unknown currency",
            id="not_found"
        )

        with self.coinBase_config() as fp:
            connect = coinbase.CoinBaseConnect(fp.name) # Initialization should pass with default mocks
            # The @manage_exception decorator in CoinBaseConnect should catch APIError and re-raise it
            with self.assertRaises(coinbase_error.APIError):
                connect.get_value(currency='UNKNOWN_CURRENCY')
        self.mock_client_instance.get_exchange_rates.assert_called_once_with(currency='UNKNOWN_CURRENCY')

    def test_get_value_keyerror_if_ref_missing(self):
        # Test if REF_CURRENCY ('EUR') is missing from the rates
        self.mock_client_instance.get_exchange_rates.return_value = {'currency': 'BTC', 'rates': {'USD': '51000.00'}} # EUR is missing

        with self.coinBase_config() as fp:
            connect = coinbase.CoinBaseConnect(fp.name)
            # This should result in a KeyError when accessing rates['rates'][REF_CURRENCY] ('EUR')
            # and the @manage_exception should catch and re-raise it or handle it.
            # Based on current CoinBaseConnect, it would be a KeyError inside get_value,
            # not caught by manage_exception's specific list. Let's see.
            # Update: manage_exception catches requests.exceptions.RequestException and CoinbaseError. KeyError is not listed.
            with self.assertRaises(KeyError): # Expecting direct KeyError
                 connect.get_value(currency='BTC')
        self.mock_client_instance.get_exchange_rates.assert_called_once_with(currency='BTC')

    def test_get_value_and_buy_sell_flow(self):
        with self.coinBase_config() as fp:
            connect = coinbase.CoinBaseConnect(fp.name)

            # Test get_value
            self.mock_client_instance.get_exchange_rates.return_value = {
                'currency': 'BTC',
                'rates': {'EUR': '50000.00'} # REF_CURRENCY is EUR
            }
            value = connect.get_value(currency='BTC')
            self.assertEqual(value, 50000.00)
            self.mock_client_instance.get_exchange_rates.assert_called_with(currency='BTC') # Called again

            # Test buy
            # Ensure _get_account_id finds the BTC account from setUp's default mock
            # self.mock_client_instance.get_accounts.return_value is already set in setUp

            # Mock the return value for the buy call
            mock_buy_response = MagicMock()
            mock_buy_response.amount = MagicMock(amount='0.1')
            mock_buy_response.subtotal = MagicMock(amount='4990') # Assuming this is value in REF_CURRENCY
            mock_buy_response.fee = MagicMock(amount='10')
            self.mock_client_instance.buy.return_value = mock_buy_response

            bought_amount, fee = connect.buy(amount=5000, currency='BTC', currency_value=50000.00)
            self.assertEqual(bought_amount, 0.1)
            self.assertEqual(fee, 10)
            self.mock_client_instance.buy.assert_called_once_with(
                'btc_acc_id', # From setUp mock account data
                amount=5000,
                currency='EUR', # REF_CURRENCY
                payment_method='mock_fiat_id', # From setUp mock payment method
                quote=True # Based on self.simulation = True from config
            )

            # Test sell
            # Mock the return value for the sell call
            mock_sell_response = MagicMock()
            mock_sell_response.amount = MagicMock(amount='0.1') # Amount of BTC sold
            mock_sell_response.subtotal = MagicMock(amount='5000') # Value in REF_CURRENCY before fee
            mock_sell_response.fee = MagicMock(amount='10') # Fee in REF_CURRENCY
            self.mock_client_instance.sell.return_value = mock_sell_response

            # _get_account_id will be called again for sell
            # Ensure get_accounts mock is reset if necessary, or that it can be called multiple times.
            # Default MagicMock behavior is fine for multiple calls returning same value.

            sold_value, sell_fee = connect.sell(amount=0.1, currency='BTC', currency_value=50000.00)
            self.assertEqual(sold_value, 50000.00) # connect.sell returns the currency_value passed to it
            self.assertEqual(sell_fee, 10)
            self.mock_client_instance.sell.assert_called_once_with(
                'btc_acc_id',
                amount=0.1,
                currency='BTC',
                payment_method='mock_fiat_id',
                quote=True
            )

