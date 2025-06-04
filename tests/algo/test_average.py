import unittest
import os
import json
from crypto_trading.algo.average import GuppyMMA
from crypto_trading.algo import model
from crypto_trading import config as app_config # For DB setup

# Helper to create a dummy config file for GuppyMMA tests (if needed for file-based init)
def create_dummy_guppy_config_file(config_path, short_terms, long_terms, buy, sell):
    # This helper is for algo-specific config, not the main app config for DB
    algo_config_data = {
        "GuppyMMA": {
            "short_term": short_terms,
            "long_term": long_terms,
            "buy": buy,
            "sell": sell
        }
    }
    # This is intended for when GuppyMMA itself loads a full file.
    # However, GuppyMMA's __init__ now expects its specific section or a path to a full config.
    # For simplicity, we'll assume this helper is used if testing file-based init of GuppyMMA.
    with open(config_path, 'w') as f:
        json.dump(algo_config_data, f)

class TestGuppyMMA(unittest.TestCase):
    db_config_file_path = "test_db_setup_config_guppy.json"

    @classmethod
    def setUpClass(cls):
        # Create a dummy main config file for DB initialization
        db_setup_config = {"database_file": "test_guppy_db.sqlite"} # Minimal needed for config.init -> setup_db
        with open(cls.db_config_file_path, 'w') as f:
            json.dump(db_setup_config, f)

        try:
            app_config.init(cls.db_config_file_path) # Initialize DB connection
            model.create() # Create tables
        except Exception as e:
            print(f"ERROR during setUpClass in TestGuppyMMA: {e}")
            raise

    @classmethod
    def tearDownClass(cls):
        try:
            model.reset() # Reset/clear tables if possible
        except Exception as e:
            print(f"ERROR during tearDownClass in TestGuppyMMA: {e}")

        if os.path.exists(cls.db_config_file_path):
            os.remove(cls.db_config_file_path)
        if os.path.exists("test_guppy_db.sqlite"): # Clean up the dummy DB file
            os.remove("test_guppy_db.sqlite")

    def setUp(self):
        self.algo_specific_config_file_path = "test_guppy_algo_specific_config.json"
        if os.path.exists(self.algo_specific_config_file_path):
            os.remove(self.algo_specific_config_file_path)

        self.default_short = GuppyMMA.SHORT_TERM_DFT[:]
        self.default_long = GuppyMMA.LONG_TERM_DFT[:]
        self.default_buy = len(self.default_short)
        self.default_sell = len(self.default_short)

    def tearDown(self):
        if os.path.exists(self.algo_specific_config_file_path):
            os.remove(self.algo_specific_config_file_path)
        # DB reset is handled in tearDownClass

    def test_initialization_with_dict(self):
        config_section = {
            "short_term": [1, 2], "long_term": [3, 4, 5],
            "buy": 1, "sell": 2
        }
        algo = GuppyMMA(config_section)
        self.assertEqual(algo.short_terms, [1, 2])
        self.assertEqual(algo.long_terms, [3, 4, 5])
        # ... (rest of assertions)
        self.assertEqual(algo.max_frequencies(), 5)
        self.assertTrue(algo.active)


    def test_initialization_with_file(self):
        # This test implies GuppyMMA's __init__ can take a file path to a full config
        # and extract its own "GuppyMMA" section.
        full_config_for_file = {
             "GuppyMMA": {"short_term": [1,2], "long_term": [3,4,5], "buy": 1, "sell": 2},
             "Bollinger": {"frequency": 20} # Other dummy data
        }
        with open(self.algo_specific_config_file_path, 'w') as f:
            json.dump(full_config_for_file, f)

        algo = GuppyMMA(self.algo_specific_config_file_path)
        self.assertEqual(algo.short_terms, [1, 2])
        self.assertEqual(algo.long_terms, [3, 4, 5])
        # ... (rest of assertions)
        self.assertEqual(algo.max_frequencies(), 5)
        self.assertTrue(algo.active)


    def test_initialization_empty_dict(self):
        algo = GuppyMMA({})
        self.assertEqual(algo.short_terms, GuppyMMA.SHORT_TERM_DFT)
        # ... (rest of assertions)
        self.assertTrue(algo.active)

    def test_initialization_missing_config_file_path(self):
        algo = GuppyMMA("non_existent_guppy_config.json")
        self.assertEqual(algo.short_terms, GuppyMMA.SHORT_TERM_DFT)
        # ... (rest of assertions)
        self.assertTrue(algo.active)


    def test_update_config(self):
        initial_config = {"short_term": [1], "long_term": [2], "buy": 0, "sell": 0}
        algo = GuppyMMA(initial_config)
        new_guppy_config_section = {
            "short_term": [10, 20], "long_term": [50, 100], "buy": 1, "sell": 1
        }
        algo.update_config(new_guppy_config_section)
        self.assertEqual(algo.short_terms, [10, 20])
        # ... (rest of assertions)
        self.assertTrue(algo.active)

    def test_update_config_partial_buy_sell(self):
        initial_config = {
            "short_term": self.default_short, "long_term": self.default_long,
            "buy": self.default_buy, "sell": self.default_sell
        }
        algo = GuppyMMA(initial_config)
        partial_update_section = {"buy": 3, "sell": 2}
        algo.update_config(partial_update_section)
        self.assertEqual(algo.buy_decision, 3)
        # ... (rest of assertions)
        self.assertTrue(algo.active)

    def test_update_config_partial_terms(self):
        initial_config = {"short_term": [1], "long_term": [2], "buy": 0, "sell": 0}
        algo = GuppyMMA(initial_config)
        partial_update_section = {"short_term": [5,6,7]}
        algo.update_config(partial_update_section)
        self.assertEqual(algo.short_terms, [5,6,7])
        # ... (rest of assertions)
        self.assertTrue(algo.active)

    def test_update_config_empty_dict(self):
        initial_config = {"short_term": [1, 2], "long_term": [3, 4, 5], "buy": 1, "sell": 2}
        algo = GuppyMMA(initial_config)
        algo.update_config({})
        self.assertEqual(algo.short_terms, GuppyMMA.SHORT_TERM_DFT)
        # ... (rest of assertions)
        self.assertTrue(algo.active)

    def test_process_not_enough_data(self):
        config_section = {"short_term": [3,5], "long_term": [10,20], "buy": 2, "sell": 2}
        algo = GuppyMMA(config_section)
        values_short = [100.0] * (max(config_section["long_term"]) - 1)
        self.assertEqual(algo.process(100.0, values_short, "BTC-USD"), 0)

    def test_process_with_enough_data_mocked_values(self):
        config_section = {"short_term": [3,5], "long_term": [8,10], "buy": 2, "sell": 2}
        algo = GuppyMMA(config_section)
        values_enough = [100.0 + i for i in range(15)]
        self.assertEqual(algo.process(100.0, values_enough, "BTC-USD"), 0)


if __name__ == '__main__':
    unittest.main()
