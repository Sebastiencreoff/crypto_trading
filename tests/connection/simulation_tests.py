from nose.tools import *
import unittest

from trading.connection import simulation


class simulationConnect_test(unittest.TestCase) :
    
    def setup(self):
        print ("SETUP!")
        
    
    def teardown(self):
        print ("TEAR DOWN!")
    
    def test_unknownCurrency(self):
    
        connect = simulation.SimulationConnect("config/connectSimu.json")

        with self.assertRaises(NameError):
            connect.get_currency(   currency = 'Unknown Currency')

        with self.assertRaises(NameError):
            connect.buy_currency(   currency = 'Unknown Currency', amt = 10)

        with self.assertRaises(NameError):
            connect.sell_currency(  currency = 'Unknown Currency', amt = 10)


    def test_execute(self):

        connect = simulation.SimulationConnect("config/connectSimu.json")

        currencyValue = connect.get_currency(   currency = 'ETH')

        self.assertTrue(currencyValue <= 100 and currencyValue >= 0)

        self.assertEqual(connect.buy_currency(   currency = 'ETH', amt = 10), ( True, 0.1 ) )
        self.assertEqual(connect.sell_currency(  currency = 'ETH', amt = 10),  True)

