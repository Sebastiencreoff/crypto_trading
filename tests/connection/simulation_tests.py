from nose.tools import *
import connection.simulation


def   simulationConnect_test(unittest.TestCase) :
    
    def setup():
        print "SETUP!"
        simulationConnect connect ("config/coinbase.json")
    
    def teardown():
        print "TEAR DOWN!"
    
    def test_unknownCurrency():
    
        with self.assertRaises(NameError):
            connect.get_currency(   currency = 'Unknown Currency')

        with self.assertRaises(NameError):
            connect.buy_currency(   currency = 'Unknown Currency', amt = 10)

        with self.assertRaises(NameError):
            connect.sell_currency(  currency = 'Unknown Currency', amt = 10)


    def test_execute():
        currencyValue = connect.get_currency(   currency = 'ETH')

        self.assertTrue(currencyValue <= 100 and currencyValue >= 0)

        self.assertEqual(connect.buy_currency(   currency = 'ETH', amt = 10), [ True, 0.1 ])
        self.assertEqual(connect.sell_currency(  currency = 'ETH', amt = 10), [ True])
    


if __name__ == "__main__":
    unittest.main()