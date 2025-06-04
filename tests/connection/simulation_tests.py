import nose.tools
import unittest

import crypto_trading.connection.simulation as simu


class SimulationConnectTest(unittest.TestCase):
    
    def setup(self):
        print('SETUP!')

    def teardown(self):
        print("TEAR DOWN!")
    
    def test_unknownCurrency(self):

        config_dict = {}
        connect = simu.SimulationConnect(config_dict)
        connect.reset()
        with self.assertRaises(NameError):
            connect.get_currency(ref_currency='Unknown Currency')

    def test_execute(self):
        config_dict = {}
        connect = simu.SimulationConnect(config_dict)
        connect.reset()
        currency_value = connect.get_currency(ref_currency='ETH')

        self.assertTrue(currency_value <= 100 and currency_value >= 0)

        self.assertFalse(connect.in_progress())
        self.assertEqual(connect.buy_currency(amount=10), 0.1)
        self.assertTrue(connect.in_progress())
        self.assertTrue(connect.sell_currency(amount=10))

