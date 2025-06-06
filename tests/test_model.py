import unittest
from unittest.mock import patch, MagicMock, NonCallableMock
import datetime

# Assuming crypto_trading is in PYTHONPATH
from crypto_trading.model import Trading, get_portfolio_value_history
# We need sqlobject.main for connection, but we'll mostly mock SQLObject instances
# For simplicity, we'll create mock objects that behave like SQLObject results.

class TestGetPortfolioValueHistory(unittest.TestCase):

    def create_mock_trade(self, id_val, buy_date_time, sell_date_time, profit):
        # Create a mock object that quacks like a SQLObject Trading instance
        trade = NonCallableMock(spec=Trading) # Use NonCallableMock for attribute-only mocks
        trade.id = id_val
        trade.buy_date_time = buy_date_time
        trade.sell_date_time = sell_date_time
        trade.profit = profit
        return trade

    @patch('crypto_trading.model.datetime') # Mock datetime.datetime within model.py
    @patch('crypto_trading.model.Trading.select')
    def test_no_trades(self, mock_select, mock_datetime):
        mock_select.return_value = [] # No trades found for any query

        # Setup mock for datetime.datetime.now()
        mock_now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.datetime.now.return_value = mock_now

        initial_capital = 1000.0
        result = get_portfolio_value_history(initial_capital)

        # Expect one point: (mocked_current_time, initial_capital)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], (mock_now, initial_capital))

        # Check calls to Trading.select()
        # First call in get_portfolio_value_history is for all_trades_ordered
        # Second call is for completed_trades (this one might not happen if first is empty)
        self.assertTrue(mock_select.called) # At least one call for all_trades_ordered


    @patch('crypto_trading.model.Trading.select')
    def test_only_completed_trades(self, mock_select):
        initial_capital = 1000.0
        trade1_buy_time = datetime.datetime(2023, 1, 1, 10, 0, 0)
        trade1_sell_time = datetime.datetime(2023, 1, 1, 11, 0, 0)
        trade1_profit = 50.0
        mock_trade1 = self.create_mock_trade(1, trade1_buy_time, trade1_sell_time, trade1_profit)

        trade2_buy_time = datetime.datetime(2023, 1, 2, 10, 0, 0) # Bought after T1 sold
        trade2_sell_time = datetime.datetime(2023, 1, 2, 11, 0, 0)
        trade2_profit = 100.0
        mock_trade2 = self.create_mock_trade(2, trade2_buy_time, trade2_sell_time, trade2_profit)

        # Mock responses for the two select calls in the function
        # 1. For determining start_time (all_trades_ordered)
        # 2. For iterating completed_trades
        # We need to ensure the first trade (mock_trade1 here) is returned by the first select call
        # to correctly establish the initial_capital timestamp.
        # The order of completed_trades should be by sell_date_time.

        # This mock setup assumes that the same mock_select instance handles multiple calls
        # and we can vary its return value based on call order or arguments.
        # A more robust way is to use side_effect if calls are distinguishable by arguments.
        # For now, let's assume the first call to select is for all_trades_ordered (by buy_date_time, id)
        # and the second is for completed_trades (by sell_date_time).

        # Simplified: Assume the first call returns trades by buy_date_time
        # Assume the second call returns trades by sell_date_time
        # In this case, the order is the same for our test data.

        # The function calls list() on the select result, so it must be iterable.
        mock_select.side_effect = [
            [mock_trade1, mock_trade2], # For all_trades_ordered (by buy_date_time)
            [mock_trade1, mock_trade2]  # For completed_trades (by sell_date_time)
        ]

        result = get_portfolio_value_history(initial_capital)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], (trade1_buy_time, initial_capital))
        self.assertEqual(result[1], (trade1_sell_time, initial_capital + trade1_profit))
        self.assertEqual(result[2], (trade2_sell_time, initial_capital + trade1_profit + trade2_profit))

    @patch('crypto_trading.model.datetime') # Mock datetime.datetime within model.py
    @patch('crypto_trading.model.Trading.select')
    def test_trades_no_buy_date_for_first_trade_defaults_to_now(self, mock_select, mock_datetime):
        initial_capital = 1000.0
        # First trade by ID/order has no buy_date_time
        mock_trade_no_buy_date = self.create_mock_trade(1, None, datetime.datetime(2023,1,1,11,0,0), 50.0)

        mock_now = datetime.datetime(2023, 1, 1, 10, 0, 0) # Mocked current time
        mock_datetime.datetime.now.return_value = mock_now

        mock_select.side_effect = [
            [mock_trade_no_buy_date], # For all_trades_ordered
            [mock_trade_no_buy_date]  # For completed_trades (if it even gets this far with None buy_date)
        ]

        result = get_portfolio_value_history(initial_capital)

        # Expect the start_time for initial capital to be mock_now
        self.assertEqual(len(result), 2) # Initial point + 1 completed trade
        self.assertEqual(result[0], (mock_now, initial_capital))
        self.assertEqual(result[1], (mock_trade_no_buy_date.sell_date_time, initial_capital + mock_trade_no_buy_date.profit))

if __name__ == '__main__':
    unittest.main()
