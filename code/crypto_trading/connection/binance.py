import os
import logging

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException

from crypto_trading.connection.connection import Connect

logger = logging.getLogger(__name__)


class BinanceConnect(Connect):
    def __init__(self, exchange_config=None): # exchange_config can be an ExchangeConfig object or dict
        """
        Initializes the BinanceConnect client.
        API key and secret are sourced from environment variables:
        BINANCE_API_KEY and BINANCE_API_SECRET.
        Other settings like 'simulation' mode can be passed via exchange_config.
        :param exchange_config: Optional. An object or dict containing non-secret
                                 exchange parameters, e.g., from AppConfig.exchanges.
        """
        self.api_key = os.environ.get("BINANCE_API_KEY")
        self.api_secret = os.environ.get("BINANCE_API_SECRET")

        if not self.api_key or not self.api_secret:
            logger.error("BINANCE_API_KEY and/or BINANCE_API_SECRET environment variables not set.")
            raise ValueError("Missing Binance API key/secret in environment variables.")

        try:
            self.client = Client(self.api_key, self.api_secret)
            # TODO: Handle 'simulation' mode or other settings from exchange_config if provided
            # For example, if exchange_config has 'extra_settings':
            # if exchange_config and hasattr(exchange_config, 'extra_settings') and exchange_config.extra_settings:
            #     if exchange_config.extra_settings.get("simulation", False):
            #         self.client.API_URL = Client.API_TESTNET_URL # Example for testnet
            #         logger.info("Binance client initialized in TESTNET/SIMULATION mode.")
            #     else:
            #         logger.info("Binance client initialized in LIVE mode.")
            # else:
            #     logger.info("Binance client initialized (defaulting to LIVE mode).")
            logger.info("Binance client initialized successfully.")

        except Exception as e:
            logger.error(f"Error initializing Binance client: {e}")
            raise

    def get_value(self, currency):
        """
        Gets the current price of a currency symbol.
        :param currency: Currency symbol (e.g., 'BTCUSDT').
        :return: Price of the currency.
        """
        try:
            ticker = self.client.get_symbol_ticker(symbol=currency)
            price = float(ticker['price'])
            logger.info(f"Current price of {currency}: {price}")
            return price
        except BinanceAPIException as e:
            logger.error(f"Binance API exception while fetching price for {currency}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching price for {currency}: {e}")
            raise

    def buy(self, amount, currency, currency_value):
        """
        Executes a market buy order.
        :param amount: Amount in quote currency (e.g., USDT).
        :param currency: Currency symbol (e.g., 'BTCUSDT').
        :param currency_value: Current price (fetched again internally for safety).
        :return: Quantity of the base asset bought and the fee.
        """
        try:
            # Fetch the latest price for safety, though currency_value is passed
            current_price = self.get_value(currency)
            quantity_to_buy = amount / current_price
            # Binance uses quantity of the base asset for market buy orders.
            # We need to calculate the quantity based on the amount in quote currency.
            # For simplicity, we'll use the provided currency_value, but it's better to fetch fresh price.
            # However, the API expects quantity of the base asset.
            # Let's adjust to buy with quoteOrderQty for precision with fiat amount

            logger.info(f"Attempting to buy {currency} with {amount} USDT.")
            order = self.client.order_market_buy(
                symbol=currency,
                quoteOrderQty=amount  # Amount in quote currency (USDT)
            )
            logger.info(f"Buy order successful: {order}")

            quantity_bought = float(order['executedQty'])

            # Calculate fees
            fee = 0
            for f in order['fills']:
                fee += float(f['commission'])
                # Assuming commissionAsset is the base currency, if not, conversion is needed
                # For simplicity, we sum up commission, assuming it's in the base asset.
                # Binance fees can be complex (BNB discounts, etc.)

            logger.info(f"Bought {quantity_bought} of {currency.replace('USDT', '')}, Fee: {fee}")
            return quantity_bought, fee

        except BinanceAPIException as e:
            logger.error(f"Binance API exception during buy order for {currency}: {e}")
            raise
        except BinanceOrderException as e:
            logger.error(f"Binance order exception during buy order for {currency}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during buy order for {currency}: {e}")
            raise

    def sell(self, amount, currency, currency_value):
        """
        Executes a market sell order.
        :param amount: Amount in base currency (e.g., BTC quantity).
        :param currency: Currency symbol (e.g., 'BTCUSDT').
        :param currency_value: Current price.
        :return: Total quote currency received from the sale and the fee.
        """
        try:
            logger.info(f"Attempting to sell {amount} of {currency.replace('USDT','')} at {currency_value}")
            order = self.client.order_market_sell(
                symbol=currency,
                quantity=amount
            )
            logger.info(f"Sell order successful: {order}")

            total_quote_received = float(order['cummulativeQuoteQty'])

            fee = 0
            # Calculate fees (similar to buy, can be complex)
            for f in order['fills']:
                fee += float(f['commission'])
                # Assuming commissionAsset is the quote currency for sells.

            logger.info(f"Sold {amount} of {currency.replace('USDT','')}. Received: {total_quote_received} USDT, Fee: {fee}")
            return total_quote_received, fee

        except BinanceAPIException as e:
            logger.error(f"Binance API exception during sell order for {currency}: {e}")
            raise
        except BinanceOrderException as e:
            logger.error(f"Binance order exception during sell order for {currency}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during sell order for {currency}: {e}")
            raise

    def get_all_values(self):
        # TODO: Implement if needed, or raise NotImplementedError
        logger.warning("get_all_values is not implemented for BinanceConnect")
        raise NotImplementedError("get_all_values is not implemented for BinanceConnect")

    def get_value_specific(self, list_cur):
        # TODO: Implement if needed, or raise NotImplementedError
        logger.warning("get_value_specific is not implemented for BinanceConnect")
        raise NotImplementedError("get_value_specific is not implemented for BinanceConnect")

    def get_balance(self):
        # TODO: Implement if needed, or raise NotImplementedError
        logger.warning("get_balance is not implemented for BinanceConnect")
        raise NotImplementedError("get_balance is not implemented for BinanceConnect")

    def get_all_transactions(self):
        # TODO: Implement if needed, or raise NotImplementedError
        logger.warning("get_all_transactions is not implemented for BinanceConnect")
        raise NotImplementedError("get_all_transactions is not implemented for BinanceConnect")

    def get_transaction_details(self, transaction_id):
        # TODO: Implement if needed, or raise NotImplementedError
        logger.warning("get_transaction_details is not implemented for BinanceConnect")
        raise NotImplementedError("get_transaction_details is not implemented for BinanceConnect")

    def get_all_orders(self):
        # TODO: Implement if needed, or raise NotImplementedError
        logger.warning("get_all_orders is not implemented for BinanceConnect")
        raise NotImplementedError("get_all_orders is not implemented for BinanceConnect")

    def get_order_details(self, order_id):
        # TODO: Implement if needed, or raise NotImplementedError
        logger.warning("get_order_details is not implemented for BinanceConnect")
        raise NotImplementedError("get_order_details is not implemented for BinanceConnect")

    def cancel_order(self, order_id):
        # TODO: Implement if needed, or raise NotImplementedError
        logger.warning("cancel_order is not implemented for BinanceConnect")
        raise NotImplementedError("cancel_order is not implemented for BinanceConnect")
