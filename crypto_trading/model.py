#!/usr/bin/env python

import datetime
import logging

import sqlobject


def create():
    Trading.createTable(ifNotExists=True)


def reset(currency=None):
    """Reset database."""
    Trading.deleteMany(Trading.q.currency == currency)


def get_current_trading():
    """Get current transaction, already buy but not yet sell.

        :return : current transaction time or None
        :example :
            >>> get_current_trading()
            "2018-03-20T11:58:32.123" or None
    """
    trade = Trading.select(Trading.q.sell_date_time == None)

    if trade.count():
        logging.info('current transaction: %s', trade[0].buy_date_time)
        return trade[0]
    return None

def get_profits():
    """Get profits

        :return : profits amount
        :example :
            >>> get_profits()
            10.0
    """
    return sum( trade.profit  for trade in  Trading.select(Trading.q.profit != None ))


def get_portfolio_value_history(configured_initial_capital: float):
    """
    Calculates the portfolio value history based on closed trades.

    Args:
        configured_initial_capital: The initial capital to start the graph with.

    Returns:
        A list of (timestamp, portfolio_value) tuples.
        The first point is (earliest_buy_time, configured_initial_capital).
        Subsequent points are (sell_time, configured_initial_capital + cumulative_profit).
        Returns a list with a single point (now, configured_initial_capital) if no trades exist
        or if the first trade has no buy_date_time.
    """
    data_points = []
    cumulative_profit = 0.0

    # Attempt to find the timestamp of the very first trade action (buy)
    # Order by ID as a proxy for creation time if buy_date_time is not set on old records
    all_trades_ordered = list(Trading.select(orderBy=[Trading.q.buy_date_time, Trading.q.id]))

    start_time = datetime.datetime.now() # Default if no trades or no valid buy_date_time
    if all_trades_ordered:
        first_trade_buy_time = all_trades_ordered[0].buy_date_time
        if first_trade_buy_time:
            start_time = first_trade_buy_time

    data_points.append((start_time, configured_initial_capital))

    # Query completed trades, ordered by sell_date_time
    completed_trades = list(Trading.select(
        Trading.q.sell_date_time != None,
        orderBy=Trading.q.sell_date_time
    ))

    for trade in completed_trades:
        if trade.profit is not None:
            cumulative_profit += trade.profit
            if trade.sell_date_time: # Should always be true due to query condition
                 data_points.append((trade.sell_date_time, configured_initial_capital + cumulative_profit))
            else:
                # This case should ideally not be reached
                logging.warning(f"Trade with ID {trade.id} marked complete but missing sell_date_time, skipping for graph.")

    return data_points


class Trading(sqlobject.SQLObject):
    buy_date_time = sqlobject.col.DateTimeCol(default=None)
    buy_value = sqlobject.col.FloatCol()
    buy_value_fee = sqlobject.col.FloatCol(default=None)
    currency_buy_amt = sqlobject.col.FloatCol(default=None)
    currency_buy_value = sqlobject.col.FloatCol(default=None)
    currency_sell_value = sqlobject.col.FloatCol(default=None)
    currency = sqlobject.col.StringCol()
    profit = sqlobject.col.FloatCol(default=None)
    sell_value_fee = sqlobject.col.FloatCol(default=None)
    sell_date_time = sqlobject.col.DateTimeCol(default=None)

    def save_buy(self, currency_amt, ref_fee_amt):
        """save buy

            :param currency_amt: value of the currency bought
            :param ref_fee_amt: fee amount in Euros

            :return : self
            :example :
                >>> save_buy(currency_amt='10',  ref_fee_amt='1')
               trading.model.Transaction or None
        """
        if not currency_amt and not ref_fee_amt:
            logging.warning('Error on buying {} bought at {}'.format(
                self.buy_value, self.buy_date_time))
            return None
        self.buy_date_time = datetime.datetime.now()
        self.buy_value_fee = ref_fee_amt
        self.currency_buy_amt = currency_amt
        self.currency_buy_value = self.buy_value / self.currency_buy_amt
        return self

    def save_sell(self, currency_sell_value, sell_value_fee):
        """Update transaction value into db.

            :param currency_sell_value: currency sell in Eur
            :param sell_value_fee: fee amount in Eur

            :return : self
            :example :
                >>> sell(sell_value = '100',
                >>>      sell_value_fee='1.99')
                trading.model.Transaction or None
        """
        if not currency_sell_value and not sell_value_fee:
            logging.critical('Error on selling {} bought at {}'.format(
                self.buy_value, self.buy_date_time))
            return None

        self.profit = (currency_sell_value * self.currency_buy_amt
                     - self.buy_value
                     - self.buy_value_fee
                     - sell_value_fee)

        self.currency_sell_value = currency_sell_value
        self.sell_value_fee = sell_value_fee
        self.sell_date_time = datetime.datetime.now()
        logging.warning('Closing trading with %s profit', self.profit)
        return self
