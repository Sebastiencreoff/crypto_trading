#!/usr/bin/env python
import datetime
import os

import nose.tools
import unittest

import trading.connection.dbtransaction


class DbTransaction_test(unittest.TestCase):
    
    def setup(self):
        print('SETUP!')
    
    def teardown(self):
        print('TEAR DOWN!')
    
    def test_transaction(self):

        db = trading.connection.dbtransaction.DbTransaction('BTC')

        db.reset()
        
        self.assertEqual(db.get_current_transaction(), None)

        buy_date = db.buy(currency_amt=1, ref_amt=10, ref_fee_amt=0.1)
        self.assertIsNotNone(buy_date)
        self.assertTrue(datetime.datetime.strptime(
            buy_date,
            trading.connection.dbtransaction.DbTransaction.DATE_TIME_FORMAT)
        )

        self.assertEqual(db.get_current_transaction()['buy_date_time'], buy_date)

        self.assertTrue(db.sell(buy_date_time=buy_date, sell_value=10))
        self.assertEqual(db.get_current_transaction(), None)

    def test_errorOnDoubleTransaction(self):

        db = trading.connection.dbtransaction.DbTransaction('BTC')
        
        db.reset()

        self.assertEqual(db.get_current_transaction(), None)

        buy_date = db.buy(currency_amt=1, ref_amt=10, ref_fee_amt=0.1)
        self.assertIsNotNone(buy_date)
        self.assertTrue(datetime.datetime.strptime(
            buy_date,
            trading.connection.dbtransaction.DbTransaction.DATE_TIME_FORMAT)
        )
        self.assertEqual(db.get_current_transaction()['buy_date_time'], buy_date)
        self.assertTrue(db.sell(buy_date_time=buy_date, sell_value=10))
        self.assertEqual(db.get_current_transaction(), None)
