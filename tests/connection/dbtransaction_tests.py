from nose.tools import *
import unittest

import os

import datetime

from trading.connection import dbtransaction


class dbTransaction_test(unittest.TestCase) :
    
    def setup(self):
        print ("SETUP!")
    
    def teardown(self):
        print ("TEAR DOWN!")
    
    def test_transaction(self):

        db = dbtransaction.DbTransaction("db", "BTC")

        db.reset()
        
        self.assertFalse(db.getCurrentTransaction())

        (result, buyDate) = db.buy(value = 1, buyValue = 10)
        self.assertTrue(result)
        self.assertTrue(datetime.datetime.strptime(buyDate, "%Y-%m-%dT%H:%M:%S.%f"))

        self.assertEqual( db.getCurrentTransaction(), (True, buyDate) )

        self.assertTrue(db.sell(buyDateTime = buyDate, sellValue = 10))
        self.assertFalse( db.getCurrentTransaction() )
        self.assertFalse(db.sell(buyDateTime = buyDate, sellValue = 10))

    def test_errorOnDoubleTransaction(self):

        db =  dbtransaction.DbTransaction("db", "BTC")
        
        db.reset()

        self.assertFalse(db.getCurrentTransaction())
        
        (result, buyDate) = db.buy(value = 1, buyValue = 10)
        self.assertTrue(result)
        self.assertTrue(datetime.datetime.strptime(buyDate, "%Y-%m-%dT%H:%M:%S.%f"))

        self.assertEqual( db.getCurrentTransaction(), (True, buyDate) )
        
        self.assertFalse(db.buy(value = 1, buyValue = 10))
        self.assertFalse(db.buy(value = 1, buyValue = 10))

        self.assertTrue(db.sell(buyDateTime = buyDate, sellValue = 10))
        self.assertFalse( db.getCurrentTransaction() )
        self.assertFalse(db.sell(buyDateTime = buyDate, sellValue = 10))