from nose.tools import *
import unittest

from trading.connection import dbTransaction


class dbTransaction_test(unittest.TestCase) :
    
    def setup(self):
        print ("SETUP!")
        
    
    def teardown(self):
        print ("TEAR DOWN!")
    
    def test_transaction(self):
    
        pass

    def test_getLastTransaction(self):

        pass

    def test_errorOnDoubleTransaction(self):

        pass

    def test_errorOnDoubleSell(self):

        pass