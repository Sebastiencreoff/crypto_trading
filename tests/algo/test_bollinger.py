import unittest
import os
import json
from crypto_trading.algo.bollinger import Bollinger
from crypto_trading.algo import model
from crypto_trading import config as app_config # For DB setup

# Helper to create a dummy config file for Bollinger tests (if needed for file-based init)
def create_dummy_bollinger_config_file(config_path, freq=20):
    # This helper is for algo-specific config, not the main app config for DB
    algo_config_data = {
        "Bollinger": {
            "frequency": freq
        }
    }
    with open(config_path, 'w') as f:
        json.dump(algo_config_data, f)

class TestBollinger(unittest.TestCase):
    db_config_file_path = "test_db_setup_config_bollinger.json"

    @classmethod
    def setUpClass(cls):
        db_setup_config = {"database_file": "test_bollinger_db.sqlite"}
        with open(cls.db_config_file_path, 'w') as f:
            json.dump(db_setup_config, f)

        try:
            app_config.init(cls.db_config_file_path)
            model.create()
        except Exception as e:
            print(f"ERROR during setUpClass in TestBollinger: {e}")
            raise

    @classmethod
    def tearDownClass(cls):
        try:
            model.reset()
        except Exception as e:
            print(f"ERROR during tearDownClass in TestBollinger: {e}")

        if os.path.exists(cls.db_config_file_path):
            os.remove(cls.db_config_file_path)
        if os.path.exists("test_bollinger_db.sqlite"):
            os.remove("test_bollinger_db.sqlite")

    def setUp(self):
        self.algo_specific_config_file_path = "test_bollinger_algo_specific_config.json"
        if os.path.exists(self.algo_specific_config_file_path):
            os.remove(self.algo_specific_config_file_path)

    def tearDown(self):
        if os.path.exists(self.algo_specific_config_file_path):
            os.remove(self.algo_specific_config_file_path)


    def test_initialization_with_dict(self):
        config_section = {"frequency": 25}
        algo = Bollinger(config_section)
        self.assertEqual(algo.frequency, 25)
        self.assertEqual(algo.max_frequencies(), 25)

    def test_initialization_with_file(self):
        # This test implies Bollinger's __init__ can take a file path to a full config
        full_config_for_file = {
            "Bollinger": {"frequency": 22},
            "GuppyMMA": {} # Other dummy data
        }
        with open(self.algo_specific_config_file_path, 'w') as f:
            json.dump(full_config_for_file, f)

        algo = Bollinger(self.algo_specific_config_file_path)
        self.assertEqual(algo.frequency, 22)
        self.assertEqual(algo.max_frequencies(), 22)

    def test_initialization_empty_dict(self):
        algo = Bollinger({})
        self.assertEqual(algo.frequency, Bollinger.DEFAULT_FREQUENCY)
        self.assertEqual(algo.max_frequencies(), Bollinger.DEFAULT_FREQUENCY)

    def test_initialization_missing_config_file_path(self):
        algo = Bollinger("non_existent_bollinger_config.json")
        self.assertEqual(algo.frequency, Bollinger.DEFAULT_FREQUENCY)
        self.assertEqual(algo.max_frequencies(), Bollinger.DEFAULT_FREQUENCY)

    def test_update_config(self):
        initial_config_section = {"frequency": 20}
        algo = Bollinger(initial_config_section)

        new_bollinger_config_section = {"frequency": 30}
        algo.update_config(new_bollinger_config_section)

        self.assertEqual(algo.frequency, 30)
        self.assertEqual(algo.max_frequencies(), 30)

    def test_update_config_missing_value(self):
        initial_config_section = {"frequency": 20}
        algo = Bollinger(initial_config_section)

        algo.update_config({})

        self.assertEqual(algo.frequency, Bollinger.DEFAULT_FREQUENCY)
        self.assertEqual(algo.max_frequencies(), Bollinger.DEFAULT_FREQUENCY)

    def test_process_not_enough_data(self):
        config_section = {"frequency": 20}
        algo = Bollinger(config_section)
        values_short = [100.0] * 19
        self.assertEqual(algo.process(100.0, values_short, "BTC-USD"), 0)

    def test_process_with_enough_data_mocked_values(self):
        config_section = {"frequency": 3}
        algo = Bollinger(config_section)
        values_enough = [100.0, 101.0, 100.0, 102.0, 99.0]
        self.assertEqual(algo.process(99.0, values_enough, "BTC-USD"), 0)


if __name__ == '__main__':
    unittest.main()
