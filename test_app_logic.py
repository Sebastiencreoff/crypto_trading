import json
import os
import sys

# Add the parent directory to sys.path to allow importing app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    # Attempt to import functions from app.py
    # Assuming app.py is in the parent directory relative to where this script might be run from
    # Or if app.py is in the same directory (e.g. if test_app_logic.py is in root)
    from app import load_config, save_config
except ImportError:
    print("Error: Could not import load_config and save_config from app.py.")
    print("Ensure app.py is in the correct location and sys.path is configured appropriately.")
    print(f"Current sys.path: {sys.path}")
    print(f"Current working directory: {os.getcwd()}")
    # Try a direct import if the script is in the root
    try:
        from app import load_config, save_config
        print("Successful direct import.")
    except ImportError as e:
         print(f"Direct import failed: {e}")
         sys.exit(1)


# --- Test Configuration ---
CONFIG_DIR = "config" # Ensure this matches app.py

def read_json_file(filepath):
    """Helper to read a JSON file directly."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def print_test_step(message):
    print(f"\n--- {message} ---")

def assert_dict_contains(expected, actual, context=""):
    """
    Checks if the 'actual' dictionary contains all key-value pairs from 'expected'.
    Allows 'actual' to have more keys than 'expected'.
    Recursively checks nested dictionaries.
    """
    for key, expected_value in expected.items():
        if key not in actual:
            print(f"FAIL {context}: Expected key '{key}' not found in actual.")
            return False
        actual_value = actual[key]
        if isinstance(expected_value, dict):
            if not isinstance(actual_value, dict):
                print(f"FAIL {context}: Expected dict for key '{key}', got {type(actual_value)}.")
                return False
            if not assert_dict_contains(expected_value, actual_value, context=f"{context} -> {key}"):
                return False
        elif isinstance(expected_value, list) and isinstance(actual_value, list):
            # Simple list comparison for this test suite, assuming order matters for short/long term periods
            if expected_value != actual_value:
                print(f"FAIL {context}: List mismatch for key '{key}'. Expected: {expected_value}, Actual: {actual_value}")
                return False
        elif expected_value != actual_value:
            # Handle float comparison with a tolerance if necessary, but direct for now
            if isinstance(expected_value, float) and isinstance(actual_value, float):
                if abs(expected_value - actual_value) > 1e-9: # Tolerance for float comparisons
                    print(f"FAIL {context}: Value mismatch for key '{key}'. Expected: {expected_value}, Actual: {actual_value}")
                    return False
            else:
                print(f"FAIL {context}: Value mismatch for key '{key}'. Expected: {expected_value}, Actual: {actual_value}")
                return False
    print(f"PASS {context}: Expected content matches actual.")
    return True


def run_tests():
    mock_session_state = {}
    results = {"passed": 0, "failed": 0}

    def record_test(success, message=""):
        if success:
            results["passed"] += 1
            # print(f"PASS: {message}")
        else:
            results["failed"] += 1
            print(f"FAIL: {message}")
        return success

    # --- Phase 1: Initial Load Test ---
    print_test_step("Phase 1: Initial Load Test")
    mock_session_state['algo_config_orig'] = load_config("algo.json")
    mock_session_state['connect_simu_config_orig'] = load_config("connectSimu.json")
    mock_session_state['trading_coinbase_config_orig'] = load_config("trading_COINBASE.json")
    mock_session_state['trading_simu_config_orig'] = load_config("trading_SIMU.json")

    initial_algo = read_json_file(os.path.join(CONFIG_DIR, "algo.json"))
    initial_connect_simu = read_json_file(os.path.join(CONFIG_DIR, "connectSimu.json"))
    initial_trading_coinbase = read_json_file(os.path.join(CONFIG_DIR, "trading_COINBASE.json"))
    initial_trading_simu = read_json_file(os.path.join(CONFIG_DIR, "trading_SIMU.json"))

    record_test(assert_dict_contains(initial_algo, mock_session_state['algo_config_orig'], "Initial algo.json load"), "Initial algo.json load")
    record_test(assert_dict_contains(initial_connect_simu, mock_session_state['connect_simu_config_orig'], "Initial connectSimu.json load"), "Initial connectSimu.json load")
    record_test(assert_dict_contains(initial_trading_coinbase, mock_session_state['trading_coinbase_config_orig'], "Initial trading_COINBASE.json load"), "Initial trading_COINBASE.json load")
    record_test(assert_dict_contains(initial_trading_simu, mock_session_state['trading_simu_config_orig'], "Initial trading_SIMU.json load"), "Initial trading_SIMU.json load")

    # --- Phase 2: Modify and Save - Simulation Mode ---
    print_test_step("Phase 2: Modify and Save - Simulation Mode")

    # Simulate UI inputs for Simulation mode
    mock_session_state['sim_file_path_input'] = "new_sim_file.csv"
    mock_session_state['sim_fee_percent_input'] = 0.0025
    mock_session_state['db_file_simu_input'] = "SIMU_updated.db" # Test plan implies some changes, let's add one
    mock_session_state['currency_simu_input'] = "ETH-USD"       # Test plan implies some changes
    mock_session_state['tx_amt_simu_input'] = 150.0             # As per test plan
    mock_session_state['delay_simu_input'] = 120               # Test plan implies some changes

    mock_session_state['guppy_short_term_str_input'] = "4,6,9,11,13,16" # Simulate change
    mock_session_state['guppy_long_term_str_input'] = "31,36,41,46,51,61" # Simulate change
    mock_session_state['guppy_buy_input'] = 3.0                 # As per test plan
    mock_session_state['guppy_sell_input'] = 3.3                # Simulate change
    mock_session_state['max_lost_percentage_input'] = 0.80      # As per test plan (float)
    mock_session_state['take_profit_percentage_input'] = 0.08   # Simulate change
    mock_session_state['bollinger_frequency_input'] = 25        # Simulate change
    mock_session_state['bollinger_std_dev_input'] = 2.5         # Simulate change

    # Construct expected connectSimu.json content
    expected_connect_simu = {**mock_session_state['connect_simu_config_orig']} # Start with original
    expected_connect_simu["file"] = "new_sim_file.csv"
    expected_connect_simu["fee_percent"] = 0.0025
    save_config("connectSimu.json", expected_connect_simu)

    # Construct expected trading_SIMU.json content
    expected_trading_simu = {**mock_session_state['trading_simu_config_orig']}
    expected_trading_simu["database_file"] = "SIMU_updated.db"
    expected_trading_simu["currency"] = "ETH-USD"
    expected_trading_simu["transactionAmt"] = 150.0
    expected_trading_simu["delay"] = 120
    # Non-UI fields should be preserved from _orig
    expected_trading_simu["connection"] = mock_session_state['trading_simu_config_orig'].get("connection")
    expected_trading_simu["connectionConfig"] = mock_session_state['trading_simu_config_orig'].get("connectionConfig")
    expected_trading_simu["algoConfig"] = mock_session_state['trading_simu_config_orig'].get("algoConfig")
    save_config("trading_SIMU.json", expected_trading_simu)

    # Construct expected algo.json content
    expected_algo_sim_save = {**mock_session_state['algo_config_orig']} # Start with original
    expected_algo_sim_save["GuppyMMA"] = {
        "short_term": [4,6,9,11,13,16],
        "long_term": [31,36,41,46,51,61],
        "buy": 3.0,
        "sell": 3.3
    }
    expected_algo_sim_save["maxLost"] = {
        **expected_algo_sim_save.get("maxLost", {}), # Preserve other maxLost fields
        "percentage": 0.80
    }
    expected_algo_sim_save["takeProfit"] = {
        **expected_algo_sim_save.get("takeProfit", {}),
        "percentage": 0.08
    }
    expected_algo_sim_save["Bollinger"] = {
        **expected_algo_sim_save.get("Bollinger", {}),
        "frequency": 25,
        "std_dev": 2.5
    }
    save_config("algo.json", expected_algo_sim_save)

    # Verify files after Simulation Mode save
    current_connect_simu = read_json_file(os.path.join(CONFIG_DIR, "connectSimu.json"))
    current_trading_simu = read_json_file(os.path.join(CONFIG_DIR, "trading_SIMU.json"))
    current_algo = read_json_file(os.path.join(CONFIG_DIR, "algo.json"))
    original_trading_coinbase = read_json_file(os.path.join(CONFIG_DIR, "trading_COINBASE.json")) # Should be unchanged

    record_test(assert_dict_contains(expected_connect_simu, current_connect_simu, "connectSimu.json after Sim save"), "connectSimu.json after Sim save")
    record_test(assert_dict_contains(expected_trading_simu, current_trading_simu, "trading_SIMU.json after Sim save"), "trading_SIMU.json after Sim save")
    record_test(assert_dict_contains(expected_algo_sim_save, current_algo, "algo.json after Sim save"), "algo.json after Sim save")
    record_test(assert_dict_contains(initial_trading_coinbase, original_trading_coinbase, "trading_COINBASE.json unchanged after Sim save"), "trading_COINBASE.json unchanged after Sim save")

    # Update _orig states to reflect the save
    mock_session_state['algo_config_orig'] = current_algo
    mock_session_state['connect_simu_config_orig'] = current_connect_simu
    mock_session_state['trading_simu_config_orig'] = current_trading_simu

    # --- Phase 3: Modify and Save - Coinbase Mode ---
    print_test_step("Phase 3: Modify and Save - Coinbase Mode")

    # Simulate UI inputs for Coinbase mode
    mock_session_state['db_file_coinbase_input'] = "COINBASE_updated.db"
    mock_session_state['currency_coinbase_input'] = "ETH-USD" # Test plan: "ETH" - assuming "ETH-USD"
    mock_session_state['tx_amt_coinbase_input'] = 250.0      # As per test plan
    mock_session_state['delay_coinbase_input'] = 240

    # New algo changes for this save step
    # GuppyMMA values should persist from the previous save (Sim mode) if not changed here
    # maxLost percentage should persist from Sim mode save
    mock_session_state['bollinger_frequency_input'] = 300 # As per test plan
    # Let's assume other algo inputs are not touched, so they should use values from previous algo save.

    # Construct expected trading_COINBASE.json
    expected_trading_coinbase = {**mock_session_state['trading_coinbase_config_orig']} # Start with its original before this test run
    expected_trading_coinbase["database_file"] = "COINBASE_updated.db"
    expected_trading_coinbase["currency"] = "ETH-USD"
    expected_trading_coinbase["transactionAmt"] = 250.0
    expected_trading_coinbase["delay"] = 240
    # Preserve non-UI fields from its original state
    expected_trading_coinbase["connection"] = initial_trading_coinbase.get("connection") # Use the very initial one
    expected_trading_coinbase["connectionConfig"] = initial_trading_coinbase.get("connectionConfig")
    expected_trading_coinbase["algoConfig"] = initial_trading_coinbase.get("algoConfig")
    save_config("trading_COINBASE.json", expected_trading_coinbase)

    # Construct expected algo.json content for Coinbase save
    # It should build upon the algo state from the *previous* save (expected_algo_sim_save)
    expected_algo_coinbase_save = {**expected_algo_sim_save}
    expected_algo_coinbase_save["Bollinger"] = {
        **expected_algo_sim_save.get("Bollinger", {}), # Preserve other Bollinger fields
        "frequency": 300
    }
    # Ensure other parts of algo.json like GuppyMMA and maxLost reflect changes from Sim mode save
    save_config("algo.json", expected_algo_coinbase_save)

    # Verify files after Coinbase Mode save
    current_trading_coinbase = read_json_file(os.path.join(CONFIG_DIR, "trading_COINBASE.json"))
    current_algo_after_coinbase = read_json_file(os.path.join(CONFIG_DIR, "algo.json"))
    # connectSimu and trading_SIMU should be as they were after the Sim mode save
    connect_simu_after_sim_save = read_json_file(os.path.join(CONFIG_DIR, "connectSimu.json"))
    trading_simu_after_sim_save = read_json_file(os.path.join(CONFIG_DIR, "trading_SIMU.json"))

    record_test(assert_dict_contains(expected_trading_coinbase, current_trading_coinbase, "trading_COINBASE.json after Coinbase save"), "trading_COINBASE.json after Coinbase save")
    record_test(assert_dict_contains(expected_algo_coinbase_save, current_algo_after_coinbase, "algo.json after Coinbase save"), "algo.json after Coinbase save")
    record_test(assert_dict_contains(expected_connect_simu, connect_simu_after_sim_save, "connectSimu.json unchanged by Coinbase save"), "connectSimu.json unchanged by Coinbase save")
    record_test(assert_dict_contains(expected_trading_simu, trading_simu_after_sim_save, "trading_SIMU.json unchanged by Coinbase save"), "trading_SIMU.json unchanged by Coinbase save")

    # Update _orig states
    mock_session_state['algo_config_orig'] = current_algo_after_coinbase
    mock_session_state['trading_coinbase_config_orig'] = current_trading_coinbase

    # --- Phase 4: Data Persistence After Save (Reload Check) ---
    print_test_step("Phase 4: Data Persistence (Reload Check after Coinbase Save)")
    # Simulate reloading configs into _orig variables
    reloaded_algo_orig = load_config("algo.json")
    reloaded_trading_coinbase_orig = load_config("trading_COINBASE.json")

    # These should match the state after the last save (Coinbase save)
    record_test(assert_dict_contains(expected_algo_coinbase_save, reloaded_algo_orig, "Reloaded algo.json matches last save"), "Reloaded algo.json")
    record_test(assert_dict_contains(expected_trading_coinbase, reloaded_trading_coinbase_orig, "Reloaded trading_COINBASE.json matches last save"), "Reloaded trading_COINBASE.json")

    print(f"\n--- Test Summary ---")
    print(f"Passed: {results['passed']}, Failed: {results['failed']}")

    return results["failed"] == 0

if __name__ == "__main__":
    # Ensure CONFIG_DIR relative to this script's location if not run from root
    # For this setup, assuming test_app_logic.py is in the root with app.py and config/
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)
        print(f"Warning: '{CONFIG_DIR}' did not exist. Created it. Config files should be present for tests.")

    if not all(os.path.exists(os.path.join(CONFIG_DIR, f)) for f in ["algo.json", "connectSimu.json", "trading_COINBASE.json", "trading_SIMU.json"]):
        print("Error: Not all base config files found. Please ensure they are created before running tests.")
        sys.exit(1)

    success = run_tests()
    if not success:
        sys.exit(1) # Exit with error code if tests failed
