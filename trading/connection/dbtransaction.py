#!/usr/bin/env python

import datetime
import logging
import sqlite3

import trading.config
import trading.utils


class Transaction(object):
    def __init__(self, transaction=None):
        self.buy_date_time = None
        self.buy_value = None
        self.buy_value_fee = None
        self.currency_buy_value = None
        self.currency_sell_value = None
        self.gain = None
        self.sell_date_time = None
        self.sell_value_fee = None

        if transaction:
            self.buy_date_time = transaction[0]
            if transaction[1]:
                self.buy_value = float(transaction[1])
            if transaction[2]:
                self.buy_value_fee = float(transaction[2])
            if transaction[3]:
                self.currency_buy_value = float(transaction[3])
            if transaction[4]:
                self.currency_sell_value = float(transaction[4])
            if transaction[5]:
                self.gain = float(transaction[5])
            self.sell_date_time = transaction[6]
            if transaction[7]:
                self.sell_value_fee = float(transaction[7])


class DbTransaction:
    """Db used to store transaction data."""

    DATE_TIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'

    def __init__(self, currency):
        """Class initialisation.

            :param currency: currency managed inside the database
        """
        self.db_file = '{}.sqlite'.format(trading.config.conf.database_file)
        self.name = 'Transaction_' + currency

        self.cnx = sqlite3.connect(self.db_file, check_same_thread=False)
        self.cursor = self.cnx.cursor()

        self.cursor.execute(
            'CREATE table IF NOT EXISTS ' + self.name +
            '(buy_date_time DATETIME,'
            ' buy_value REAL,'
            ' buy_value_fee REAL,'
            ' currency_buy_value REAL,'
            ' currency_sell_value REAL,'
            ' gain REAL,'
            ' sell_date_time DATETIME,'
            ' sell_value_fee REAL)'
            )
        self.query = None
        self.cnx.commit()

    @trading.utils.database_query
    def buy(self, currency_amt, ref_amt, ref_fee_amt):
        """Add transaction value into db.

            :param currency_amt: value of the currency bought
            :param ref_amt: value of the currency bought in Euros
            :param ref_fee_amt: fee in Euro of the transaction

            :return : transaction time id or None if failed
            :example :
                >>> buy(currency_amt='10', ref_amt='100', ref_fee_amt='1')
                time of the transaction or None
        """

        current_time = datetime.datetime.now()
        self.query = ('INSERT INTO {} (currency_buy_value, buy_value, '
                      'buy_value_fee, buy_date_time) '
                      'VALUES (?,?,?,?)'.format(self.name))
        self.cursor.execute(
            self.query,
            [currency_amt, ref_amt, ref_fee_amt, current_time])
        
        self.cnx.commit()

        logging.info('transaction saved: {}'.format(current_time))
        return current_time.strftime(self.DATE_TIME_FORMAT)

    @trading.utils.database_query
    def sell(self,  transaction,
             currency_sell_value, sell_value_fee,
             sell_date_time='NOW'):
        """Update transaction value into db.

            :param buy_date_time: currency buy time
            :param sell_value: value of the currency sell in Euros
            :param sell_date_time: currency sell time

            :return : boolean which indicates function status
            :example :
                >>> sell(buy_date_time= time, sell_value = '100',
                >>>      sell_value_fee='1.99',
                >>>      sell_date_time='NOW')
                boolean True or False
        """

        self.query = ('UPDATE {} SET gain = ?, '
                      'currency_sell_value = ?, '
                      'sell_value_fee = ?, '
                      'sell_date_time = ? '
                      'WHERE buy_date_time = ? '.format(self.name))

        gain = ((float(currency_sell_value) - transaction.currency_buy_value)
                * transaction.buy_value
                - transaction.buy_value_fee
                - float(sell_value_fee))

        self.cursor.execute(
            self.query,
            [gain, currency_sell_value, sell_value_fee, sell_date_time,
             transaction.buy_date_time])
        
        logging.info('transaction saved: %s', sell_date_time)
        self.cnx.commit()

        return gain

    @trading.utils.database_query
    def get_current_transaction(self):
        """Get current transaction, already buy but not yet sell.

            :return : current transaction time or None
            :example :
                >>> get_current_transaction()
                 "2018-03-20T11:58:32.123" or None
        """
        self.query = ('SELECT '
                      'buy_date_time, buy_value, buy_value_fee, '
                      'currency_buy_value, currency_sell_value, '
                      'gain, '
                      'sell_date_time, sell_value_fee FROM {} '
                      'WHERE sell_date_time is null'.format(self.name))

        self.cursor.execute(self.query)

        # fetchone return a tuple
        transaction = None
        datas = self.cursor.fetchone()
        if datas:
            transaction = Transaction(datas)
            logging.info('current transaction: %s', transaction.buy_date_time)
        return transaction

    def reset(self):
        """Reset database."""
        self.cursor.execute('DELETE FROM ' + self.name)
        self.cnx.commit()
