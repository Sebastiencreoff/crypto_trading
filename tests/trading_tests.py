#!/usr/bin/env python3
import unittest
from unittest.mock import patch, MagicMock, PropertyMock, call
import time # Will be mocked

# EndOfProcess is defined in simulation.py
from crypto_trading.connection.simulation import EndOfProcess
from crypto_trading.trading import Trading

# Mock paths based on the project structure observed
CONFIG_MODULE_PATH = 'crypto_trading.config'
CONNECTION_MODULE_PATH = 'crypto_trading.connection'
ALGO_MODULE_PATH = 'crypto_trading.algo'
MODEL_MODULE_PATH = 'crypto_trading.model'
TRADING_MODULE_PATH = 'crypto_trading.trading' # For time.sleep

class TestTrading(unittest.TestCase):

    def setUp(self):
        # Start patches and keep references to the mock objects
        self.patchers = [
            patch(f'{CONFIG_MODULE_PATH}.init'),
            patch(f'{CONNECTION_MODULE_PATH}.coinBase.CoinBaseConnect'),
            patch(f'{CONNECTION_MODULE_PATH}.simulation.SimulationConnect'),
            patch(f'{ALGO_MODULE_PATH}.AlgoMain'),
            patch(f'{ALGO_MODULE_PATH}.Security'),
            patch(f'{MODEL_MODULE_PATH}.create'),
            patch(f'{MODEL_MODULE_PATH}.get_current_trading'),
            patch(f'{MODEL_MODULE_PATH}.Trading'),
            patch(f'{MODEL_MODULE_PATH}.get_profits'),
            patch(f'{MODEL_MODULE_PATH}.reset'),
            patch(f'{TRADING_MODULE_PATH}.time.sleep')
        ]

        self.mock_config_init = self.patchers[0].start()
        self.mock_coinbase_connect_constructor = self.patchers[1].start()
        self.mock_sim_connect_constructor = self.patchers[2].start()
        self.mock_algo_main_constructor = self.patchers[3].start()
        self.mock_security_constructor = self.patchers[4].start()
        self.mock_model_create = self.patchers[5].start()
        self.mock_model_get_current_trading = self.patchers[6].start()
        self.mock_model_trading_constructor = self.patchers[7].start()
        self.mock_model_get_profits = self.patchers[8].start()
        self.mock_model_reset = self.patchers[9].start()
        self.mock_time_sleep = self.patchers[10].start()

        # Create a default mock config object
        self.mock_config = MagicMock()
        self.mock_config.config_file = 'dummy_config_path.json' # Or the arg to Trading()
        self.mock_config.dir_path = '/dummy/dir'
        self.mock_config.currency = 'BTC-USD'
        # Make conf.delay None or 0 to speed up tests/prevent sleep
        # Assuming 'delay' is a top-level attribute of the config object from config.init()
        # If it's nested, like self.mock_config.trading.delay, adjust accordingly.
        type(self.mock_config).delay = PropertyMock(return_value=0) # Access as property
        self.mock_config.connection = 'coinbase' # Default to coinbase for some tests
        self.mock_config.connection_config = {'api_key': 'dummy_key'} # Dummy connection_config
        self.mock_config.algo_config = {'some_algo_param': True} # Dummy algo_config

        # Configure mock_config_init to return this default mock_config
        self.mock_config_init.return_value = self.mock_config

        # Instances that will be created by Trading class
        self.mock_connect_instance = MagicMock()
        self.mock_coinbase_connect_constructor.return_value = self.mock_connect_instance
        self.mock_sim_connect_constructor.return_value = self.mock_connect_instance # Both point to same generic mock for now

        self.mock_algo_main_instance = MagicMock()
        self.mock_algo_main_constructor.return_value = self.mock_algo_main_instance

        self.mock_security_instance = MagicMock()
        self.mock_security_constructor.return_value = self.mock_security_instance

        # This mock is for the class/constructor crypto_trading.model.Trading
        # It will be used to create instances that represent a trading record from DB
        self.mock_model_trading_instance = MagicMock() # Removed spec_set=True to allow arbitrary attributes for testing
        # self.mock_model_trading_constructor.return_value = self.mock_model_trading_instance # If model.Trading() is called directly
        # Typically, get_current_trading would return an instance, so that's what we mock mostly.
        self.mock_model_get_current_trading.return_value = None # Default: no active trade

    # Test Cases will be added below

    def tearDown(self):
        for p in self.patchers:
            p.stop()

    def test_initialization_coinbase(self):
        # Access mocks via self
        mock_config_reconfigured = MagicMock() # Create a new mock for config specific to this test
        mock_config_reconfigured.delay = 0
        mock_config_reconfigured.connection = 'coinbase'
        mock_config_reconfigured.connection_config = {'coinbase_specific': True}
        mock_config_reconfigured.algo_config = {'algo_specific': True}
        mock_config_reconfigured.currency = 'BTC-TEST'
        self.mock_config_init.return_value = mock_config_reconfigured # Configure the shared mock

        trading = Trading(config_file='dummy_path_coinbase.json')

        self.mock_config_init.assert_called_once_with('dummy_path_coinbase.json')
        self.mock_coinbase_connect_constructor.assert_called_once_with(mock_config_reconfigured.connection_config)
        self.mock_sim_connect_constructor.assert_not_called()
        self.mock_algo_main_constructor.assert_called_once_with(mock_config_reconfigured.algo_config)
        self.mock_security_constructor.assert_called_once_with(mock_config_reconfigured.algo_config)
        self.mock_model_create.assert_not_called() # model.create() is called in run(), not __init__()

    def test_initialization_simulation(self):
        # Access mocks via self
        mock_config_reconfigured = MagicMock()
        mock_config_reconfigured.delay = 0
        mock_config_reconfigured.connection = 'simu'
        mock_config_reconfigured.connection_config = {'simu_specific': True}
        mock_config_reconfigured.algo_config = {'algo_specific_sim': True}
        mock_config_reconfigured.dir_path = '/simu/path'
        mock_config_reconfigured.currency = 'ETH-TEST'
        self.mock_config_init.return_value = mock_config_reconfigured

        trading = Trading(config_file='dummy_path_simu.json')

        self.mock_config_init.assert_called_once_with('dummy_path_simu.json')
        self.mock_sim_connect_constructor.assert_called_once_with(mock_config_reconfigured.connection_config, mock_config_reconfigured.dir_path)
        self.mock_coinbase_connect_constructor.assert_not_called()
        self.mock_algo_main_constructor.assert_called_once_with(mock_config_reconfigured.algo_config)
        self.mock_security_constructor.assert_called_once_with(mock_config_reconfigured.algo_config)
        self.mock_model_create.assert_not_called() # model.create() is called in run(), not __init__()

    # Test for run() method - stops on EndOfProcess
    def test_run_loop_stops_on_end_of_process(self):
        # Configure self.mock_config for this test, as it's used by Trading
        # self.mock_config is returned by self.mock_config_init by default from setUp
        # Or, reconfigure self.mock_config_init.return_value here if specific config values are needed beyond defaults
        self.mock_config.currency = "TEST-CUR" # Ensure currency is set on the shared mock_config

        # Configure the connect instance (from setUp) to raise EndOfProcess
        self.mock_connect_instance.get_value.side_effect = EndOfProcess

        # Instantiate Trading - it will use the mocks from setUp
        # The config_file path here will be passed to self.mock_config_init
        trading = Trading(config_file='dummy_path_for_run_test.json')

        # Call run()
        trading.run()

        # Assertions
        self.mock_config_init.assert_called_with('dummy_path_for_run_test.json') # Called again, or check call_count
        self.mock_model_create.assert_called_once() # model.create() is called at the start of run()
        self.mock_connect_instance.get_value.assert_called_once_with(self.mock_config.currency) # Uses self.mock_config

        # Ensure other processing methods were not called significantly or at all
        self.mock_algo_main_instance.process.assert_not_called()
        # self.mock_security_instance.process.assert_not_called() # security.process IS called
        self.mock_security_instance.buy.assert_not_called()
        self.mock_security_instance.sell.assert_not_called()
        self.mock_connect_instance.buy.assert_not_called()
        self.mock_connect_instance.sell.assert_not_called()

    def test_run_loop_buy_logic(self):
        # Configure connect.get_value to return a couple of values, then EndOfProcess
        self.mock_connect_instance.get_value.side_effect = [100, 100, EndOfProcess] # Value, then same value to trigger processing once

        # Configure config for transaction amount
        self.mock_config.transaction_amt = 500 # Example transaction amount

        # No current trading transaction initially
        self.mock_model_get_current_trading.return_value = None

        # Algo signals BUY
        self.mock_algo_main_instance.process.return_value = 1

        # Security allows BUY
        self.mock_security_instance.buy.return_value = True

        # Mock connect.buy return value
        mock_buy_return_details = ('buy_tx_id_001', 0.05) # (e.g. id, fee) - structure depends on what save_buy expects
        self.mock_connect_instance.buy.return_value = mock_buy_return_details

        # Mock the model.Trading class that is instantiated for a new transaction
        # This instance is what trans.save_buy() will be called on.
        # The mock_model_trading_constructor is for the class crypto_trading.model.Trading
        new_trade_instance_mock = MagicMock()
        new_trade_instance_mock.buy_date_time = None # Crucial for buy logic to be entered
        # The Trading class will instantiate model.Trading with buy_value and currency from config.
        # The trans object (new_trade_instance_mock) should have these attributes for the connect.buy call.
        new_trade_instance_mock.buy_value = self.mock_config.transaction_amt
        new_trade_instance_mock.currency = self.mock_config.currency # Assuming self.mock_config.currency is set as expected
        self.mock_model_trading_constructor.return_value = new_trade_instance_mock

        trading = Trading(config_file='dummy_config_buy.json')
        trading.run()

        # Assertions
        self.mock_config_init.assert_called_with('dummy_config_buy.json')
        self.mock_model_create.assert_called_once() # From run() start

        # get_value called twice before EndOfProcess
        self.assertEqual(self.mock_connect_instance.get_value.call_count, 3) # 100, 100, EndOfProcess

        # model.get_current_trading called (at least once, possibly more depending on loop structure)
        self.mock_model_get_current_trading.assert_called()

        # model.Trading constructor called for the new trade
        self.mock_model_trading_constructor.assert_called_once_with(
            buy_value=self.mock_config.transaction_amt,
            currency=self.mock_config.currency
        )

        # Algo and security processing
        self.mock_algo_main_instance.process.assert_called_once_with(
            100, # current_value
            currency=self.mock_config.currency
        )
        self.mock_security_instance.process.assert_called_once_with(
            100, # current_value
            currency=self.mock_config.currency
        )
        self.mock_security_instance.buy.assert_called_once_with(
            100, # current_value
            currency=self.mock_config.currency
        )

        # Connection buy method
        self.mock_connect_instance.buy.assert_called_once_with(
            self.mock_config.transaction_amt, # trans.buy_value
            self.mock_config.currency,        # trans.currency
            100                               # currency_value
        )

        # Save buy on the new trade instance
        new_trade_instance_mock.save_buy.assert_called_once_with(*mock_buy_return_details)

        # Ensure sell was not called
        self.mock_connect_instance.sell.assert_not_called()
        new_trade_instance_mock.save_sell.assert_not_called()

    def test_run_loop_sell_logic(self):
        # Existing active buy: model.get_current_trading() returns a transaction instance
        # This mock_model_trading_instance is from self.setUp, can be configured here
        # type() is used to mock a property on an instance mock
        type(self.mock_model_trading_instance).buy_date_time = PropertyMock(return_value='2023-01-01T10:00:00.000000')
        # Additional attributes needed by the sell logic for `trans` (self.mock_model_trading_instance)
        self.mock_model_trading_instance.currency_buy_amt = 0.5 # Example amount bought
        self.mock_model_trading_instance.currency = self.mock_config.currency # Should match config

        self.mock_model_get_current_trading.return_value = self.mock_model_trading_instance

        # Configure connect.get_value
        self.mock_connect_instance.get_value.side_effect = [110, 110, EndOfProcess] # Sell at 110

        # Algo signals SELL
        self.mock_algo_main_instance.process.return_value = -1

        # Security does NOT force sell (returns False)
        self.mock_security_instance.sell.return_value = False

        # Mock connect.sell return value
        mock_sell_return_details = ('sell_tx_id_001', 0.06) # (e.g. id, fee) - structure depends on save_sell
        self.mock_connect_instance.sell.return_value = mock_sell_return_details

        trading = Trading(config_file='dummy_config_sell.json')
        trading.run()

        # Assertions
        self.mock_config_init.assert_called_with('dummy_config_sell.json')
        self.mock_model_create.assert_called_once()

        self.assertEqual(self.mock_connect_instance.get_value.call_count, 3)
        self.mock_model_get_current_trading.assert_called() # Called in each loop iteration until trans is None

        # Algo and security processing
        self.mock_algo_main_instance.process.assert_called_once_with(110, currency=self.mock_config.currency)
        self.mock_security_instance.process.assert_called_once_with(110, currency=self.mock_config.currency)

        # security.sell is NOT called in this scenario due to short-circuiting (result < 0 is True)
        self.mock_security_instance.sell.assert_not_called()

        # Connection sell method
        self.mock_connect_instance.sell.assert_called_once_with(
            self.mock_model_trading_instance.currency_buy_amt,
            self.mock_model_trading_instance.currency,
            110 # current_value
        )

        # Save sell on the existing trade instance
        self.mock_model_trading_instance.save_sell.assert_called_once_with(*mock_sell_return_details)

        # Ensure buy was not called
        self.mock_connect_instance.buy.assert_not_called()
        # Ensure model.Trading() constructor was not called to create a new trade object
        self.mock_model_trading_constructor.assert_not_called()

    def test_run_loop_security_sell_logic(self):
        # Existing active buy
        type(self.mock_model_trading_instance).buy_date_time = PropertyMock(return_value='2023-01-01T10:00:00.000000')
        self.mock_model_trading_instance.currency_buy_amt = 0.5
        self.mock_model_trading_instance.currency = self.mock_config.currency
        self.mock_model_get_current_trading.return_value = self.mock_model_trading_instance

        # Configure connect.get_value
        self.mock_connect_instance.get_value.side_effect = [120, 120, EndOfProcess] # Price for security sell

        # Algo signals HOLD (not sell)
        self.mock_algo_main_instance.process.return_value = 0

        # Security FORCES sell (returns True)
        self.mock_security_instance.sell.return_value = True

        mock_sell_return_details = ('sell_tx_id_sec002', 0.07)
        self.mock_connect_instance.sell.return_value = mock_sell_return_details

        trading = Trading(config_file='dummy_config_sec_sell.json')
        trading.run()

        # Assertions
        self.mock_config_init.assert_called_with('dummy_config_sec_sell.json')
        self.mock_model_create.assert_called_once()

        self.assertEqual(self.mock_connect_instance.get_value.call_count, 3)
        self.mock_model_get_current_trading.assert_called()

        self.mock_algo_main_instance.process.assert_called_once_with(120, currency=self.mock_config.currency)
        self.mock_security_instance.process.assert_called_once_with(120, currency=self.mock_config.currency)

        # security.sell IS called and is the reason for the sell
        self.mock_security_instance.sell.assert_called_once_with(
            current_value=120,
            transaction=self.mock_model_trading_instance
        )

        self.mock_connect_instance.sell.assert_called_once_with(
            self.mock_model_trading_instance.currency_buy_amt,
            self.mock_model_trading_instance.currency,
            120 # current_value
        )

        self.mock_model_trading_instance.save_sell.assert_called_once_with(*mock_sell_return_details)

        self.mock_connect_instance.buy.assert_not_called()
        self.mock_model_trading_constructor.assert_not_called()

    def test_stop_method(self):
        # Make get_value simulate a few successful calls, then would continue
        # if not stopped.
        # The actual values don't matter much, just that the loop runs a few times.
        # Ensure EndOfProcess is not raised by get_value for this test.
        self.mock_connect_instance.get_value.side_effect = [100, 101, 102, 103, 104, 105]

        # Configure algo process to return a neutral signal
        self.mock_algo_main_instance.process.return_value = 0

        # Configure mock_time_sleep to call trading.stop() after a few iterations.
        # Let's say after 3 calls to sleep (which means after 3 loop iterations where processing might happen)
        stop_call_count = 3

        trading_instance_ref = [] # To store the trading instance for stop() call

        def sleep_then_stop_side_effect(delay):
            # This function will be called by time.sleep(self.conf.delay)
            # We want to call trading.stop() from here.
            # Need a reference to the trading instance.
            if self.mock_time_sleep.call_count >= stop_call_count:
                 if trading_instance_ref:
                    trading_instance_ref[0].stop()
            return None # time.sleep returns None

        self.mock_time_sleep.side_effect = sleep_then_stop_side_effect

        # Set a delay in config so time.sleep is called
        type(self.mock_config).delay = PropertyMock(return_value=0.01) # Actual delay value doesn't matter for mocked sleep

        trading = Trading(config_file='dummy_config_stop.json')
        trading_instance_ref.append(trading) # Store instance for sleep_then_stop_side_effect

        trading.run()

        # Assertions
        self.mock_config_init.assert_called_with('dummy_config_stop.json')
        # The loop should have been stopped by trading.stop()
        # get_value is called at the start of each loop.
        # If stop is called after 3 sleeps, get_value would have been called 3 or 4 times.
        # 1st call to get_value, then sleep. 2nd call, then sleep. 3rd call, then sleep (stop called). 4th call might happen.
        # The loop condition `while self.loop == 1` is checked after sleep.
        # If stop() sets self.loop = 0 during sleep, the current iteration finishes,
        # but the next one won't start.
        # So, get_value should be called `stop_call_count` times.
        self.assertEqual(self.mock_connect_instance.get_value.call_count, stop_call_count)
        self.mock_time_sleep.assert_called() # Ensure sleep was actually called
        # It's tricky to assert exact number of sleeps if loop termination is immediate
        # but it should be around stop_call_count.
        self.assertLessEqual(self.mock_time_sleep.call_count, stop_call_count)

    def test_profits_method(self):
        expected_profit = 123.45
        self.mock_model_get_profits.return_value = expected_profit

        trading = Trading(config_file='dummy_config_profits.json')
        returned_profit = trading.profits()

        self.mock_model_get_profits.assert_called_once()
        self.assertEqual(returned_profit, expected_profit)

    def test_reset_method(self):
        # self.mock_config is already set up by setUp and returned by self.mock_config_init
        # We can rely on self.mock_config.currency which is 'BTC-USD' by default from setUp.
        # If a different currency is needed for the test, reconfigure:
        # self.mock_config.currency = 'ETH-TEST'
        # Or, ensure the config object returned by self.mock_config_init has the desired currency for this test.
        # For this test, default 'BTC-USD' is fine.

        trading = Trading(config_file='dummy_config_reset.json')
        trading.reset()

        self.mock_model_reset.assert_called_once_with(self.mock_config.currency)
        self.mock_algo_main_instance.reset.assert_called_once()

if __name__ == '__main__':
    unittest.main()
