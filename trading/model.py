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


class Trading(sqlobject.SQLObject):
    buy_date_time = sqlobject.col.DateTimeCol(default=None)
    buy_value = sqlobject.col.FloatCol()
    buy_value_fee = sqlobject.col.FloatCol(default=None)
    currency_buy_value = sqlobject.col.FloatCol(default=None)
    currency = sqlobject.col.StringCol()
    currency_sell_value = sqlobject.col.FloatCol(default=None)
    gain = sqlobject.col.FloatCol(default=None)
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
        self.currency_buy_value = currency_amt
        return self

    def save_sell(self, currency_sell_value, sell_value_fee):
        """Update transaction value into db.

            :param currency_sell_value: currency sell in Eur
            :param sell_value_fee: fee amount in Eur

            :return : self
            :example :
                >>> sell(currency_sell_value = '100',
                >>>      sell_value_fee='1.99')
                trading.model.Transaction or None
        """
        if not currency_sell_value and not sell_value_fee:
            logging.critical('Error on selling {} bought at {}'.format(
                self.buy_value, self.buy_date_time))
            return None

        self.gain = ((float(currency_sell_value) - self.currency_buy_value)
                     * self.buy_value
                     - self.buy_value_fee
                     - float(sell_value_fee))

        self.currency_sell_value = currency_sell_value
        self.sell_value_fee = sell_value_fee
        self.sell_date_time = datetime.datetime.now()
        return self
