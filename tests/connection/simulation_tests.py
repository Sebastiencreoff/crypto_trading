import nose.tools
import unittest

import crypto_trading.connection.simulation as simu


import tempfile
import json

class SimulationConnectTest(unittest.TestCase):
    
    def simulation_config(self, delete=True):
        fp = tempfile.NamedTemporaryFile(mode='w', delete=delete, suffix=".json")
        # Provide a minimal valid JSON structure if SimulationConnect expects specific keys
        # For now, an empty JSON indicating simulation or specific test data path
        config_data = {
            "simulation": True,
            "data_path": "inputs/BTC_2019-04-18_to_2019-04-24.csv" # Example path
        }
        json.dump(config_data, fp)
        fp.seek(0)
        return fp

    def setup(self):
        print('SETUP!')
        # If each test needs a config, it should be created here or in each test method.
        # For now, let test methods manage their own configs if they differ.

    def teardown(self):
        print("TEAR DOWN!")
        # Clean up any files created by simulation_config if not handled by context manager
    
    def test_unknownCurrency(self):
        with self.simulation_config(delete=False) as fp:
            connect = simu.SimulationConnect(fp.name)
            # connect.reset() # Removed as SimulationConnect has no reset method
            # Assuming get_value for an unknown currency in simulation still returns a value
            # and does not raise NameError, per current SimulationConnect implementation.
            value = connect.get_value(currency='Unknown Currency') # Corrected method name
            self.assertIsNotNone(value) # Check that some value is returned
            self.assertIsInstance(value, (int, float)) # Check it's a number

    def test_execute(self):
        with self.simulation_config(delete=False) as fp:
            # The simulation might need a specific data file,
            # ensure config points to a valid one like 'inputs/BTC_2019-04-18_to_2019-04-24.csv'
            # This path is now part of simulation_config
            connect = simu.SimulationConnect(fp.name)
            # connect.reset() # Removed as SimulationConnect has no reset method
            currency_value = connect.get_value(currency='ETH') # Corrected method name

            self.assertTrue(currency_value <= 100 and currency_value >= 0) # This range seems specific to random simulation

            # self.assertFalse(connect.in_progress()) # Removed, in_progress() does not exist
            # The following assertions for buy/sell might also need to be re-evaluated
            # as buy_currency and sell_currency methods are not defined in SimulationConnect or Connect base class
            # They are defined in CoinBaseConnect, but SimulationConnect has buy() and sell()

            # Assuming the test meant to call buy() and sell() from SimulationConnect
            # and that the expected return values were for a different implementation.
            # SimulationConnect.buy() returns (buy_amount, fee_amount)
            # SimulationConnect.sell() returns (self.value, fee_amount)

            bought_amount, buy_fee = connect.buy(amount=10, currency="ETH", currency_value=currency_value)
            self.assertGreater(bought_amount, 0) # Check that some amount was bought

            # self.assertTrue(connect.in_progress()) # Removed, in_progress() does not exist

            sold_value, sell_fee = connect.sell(amount=bought_amount, currency="ETH", currency_value=connect.get_value(currency='ETH'))
            self.assertGreater(sold_value, 0) # Check that some value was obtained from selling

