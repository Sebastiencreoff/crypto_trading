#! /usr/bin/env python
import sys # Added for path modification
import os # Already imported, but ensure it's high enough for path mod
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Add parent dir to path

import copy
import json
import pathlib
import requests
import time
from datetime import datetime, timedelta
import uuid

# from phoenics import Phoenics # Commented out due to build issues
# from crypto_trading import trading # This was already fine with sys.path
from crypto_trading import trading


CONFIG = {
    "GuppyMMA" : {
        "short_term" : [3, 5, 8, 10, 12, 15],
        "long_term" : [30, 35, 40, 45, 50, 60],
        "buy" : 6,
        "sell" : 6
    },
    "maxLost" : {
        "percentage" : 3,
        "percentage_update"   : 0.25,
        "mean" : 720
    },
    "takeProfit": {
        "percentage" : 5
    },
    "Bollinger": {
        "frequency" : 100
    }
}

MAX_ITER = 100
MAX_PROFIT = 0

# Helper function for JSON loading, modifying, and saving
def _load_modify_save_json(original_path, new_path, modifications):
    try:
        with open(original_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Original JSON file not found at {original_path}")
        return False
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {original_path}")
        return False

    data.update(modifications)

    try:
        with open(new_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except IOError:
        print(f"Error: Could not write JSON to {new_path}")
        return False

def evaluate_configuration(params, bitcoin_prices):
    config = copy.deepcopy(CONFIG)
    config['GuppyMMA']['short_term'] = [int(x * params['GuppyMMA_coef'][0]) for x in config['GuppyMMA']['short_term']]
    config['GuppyMMA']['long_term'] = [int(x * params['GuppyMMA_coef'][0]) for x in config['GuppyMMA']['long_term']]
    config['GuppyMMA']['buy'] = int(params['GuppyMMA_buy'][0])
    config['GuppyMMA']['sell'] = int(params['GuppyMMA_sell'][0])
    config['maxLost']['percentage'] = int(params['maxLost_perc'][0])
    config['maxLost']['percentage_update'] = float(params['maxLost_perc_update'][0])
    config['maxLost']['mean'] = int(params['maxLost_mean'][0])
    config['takeProfit']['percentage'] = float(params['takeProfit_perc'][0])
    config['Bollinger']['frequency'] = int(params['bollinger'][0])

    pathlib.Path('./config/algo.json').write_text(json.dumps(config, indent=2))

    temp_csv_name = f"temp_bitcoin_prices_{uuid.uuid4()}.csv"
    temp_csv_path = os.path.join("config", temp_csv_name)

    temp_connect_simu_name = f"connectSimu_temp_{uuid.uuid4()}.json"
    temp_connect_simu_path = os.path.join("config", temp_connect_simu_name)

    temp_trading_simu_name = f"trading_SIMU_temp_{uuid.uuid4()}.json"
    temp_trading_simu_path = os.path.join("config", temp_trading_simu_name)

    simu = None  # Initialize simu to None

    try:
        # Write bitcoin prices to temporary CSV
        with open(temp_csv_path, 'w') as f:
            for price in bitcoin_prices:
                f.write(f"{price}\n")

        # Create temporary connectSimu_temp.json
        connect_modifications = {
            "type": "file",
            "file": temp_csv_name # Relative to config directory
        }
        if not _load_modify_save_json('config/connectSimu.json', temp_connect_simu_path, connect_modifications):
            params['profit'] = -float('inf') # Indicate error
            return params


        # Create temporary trading_SIMU_temp.json
        trading_modifications = {
            "connectionConfig": temp_connect_simu_name # Relative to config directory
        }
        if not _load_modify_save_json('config/trading_SIMU.json', temp_trading_simu_path, trading_modifications):
            params['profit'] = -float('inf') # Indicate error
            return params

        # Ensure path for trading.Trading is relative to the script location or absolute
        # Since temp_trading_simu_path is "config/temp_trading_SIMU_temp_xyz.json"
        # and the script is in "scripts/", trading.Trading expects paths relative to where it's called from or absolute.
        # If trading.Trading is called from the root, then "./config/..." is correct.
        # If called from scripts/, then "../config/..." might be needed or absolute path.
        # Given the original call was './config/trading_SIMU.json', it implies the execution context is likely the repo root.
        simu = trading.Trading(f'./{temp_trading_simu_path}')
        simu.run()
        params['profit'] = simu.profits()
    
    except Exception as e:
        print(f"An error occurred during simulation: {e}")
        params['profit'] = -float('inf') # Indicate error
    finally:
        if simu:
            simu.reset()

        # Cleanup temporary files
        for f_path in [temp_csv_path, temp_connect_simu_path, temp_trading_simu_path]:
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                except OSError as e:
                    print(f"Error deleting temporary file {f_path}: {e}")

    global MAX_PROFIT
    if params['profit'] > MAX_PROFIT:
        MAX_PROFIT = params['profit']
        print(f'MAX Profit: {params["profit"]} Config: {config}')

    print(f'Profit: {params["profit"]} Config: {config}')
    return params

def download_bitcoin_data_last_month():
    """Downloads the last 30 days of Bitcoin price data from CoinGecko."""
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"

    to_timestamp = int(time.mktime(datetime.now().timetuple()))
    from_timestamp = int(time.mktime((datetime.now() - timedelta(days=30)).timetuple()))

    params = {
        "vs_currency": "usd",
        "from": str(from_timestamp),
        "to": str(to_timestamp),
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        prices = [item[1] for item in data.get("prices", [])]
        return prices
    except requests.exceptions.RequestException as e:
        print(f"Error downloading Bitcoin data: {e}")
        return []

def main():
    phoenics = Phoenics('scripts/phoenics.json') # Restored

    # Download Bitcoin data
    bitcoin_prices = download_bitcoin_data_last_month()
    if not bitcoin_prices:
        print("Failed to download Bitcoin price data. Exiting.")
        return # Exit if no data

    if len(bitcoin_prices) == 0: # Check if prices list is empty
        print("Bitcoin price data is empty. Simulation cannot run. Exiting.")
        return

    print(f"Successfully downloaded {len(bitcoin_prices)} Bitcoin price points.")

    observations = []
    for num_iter in range(MAX_ITER):
        print(f'RUN: {num_iter}')
        # query for new parameters
        # Ensure phoenics.recommend is robust if observations is empty on first call
        params_list = phoenics.recommend(observations=observations if observations else None)
    
        # evaluate the proposed parameters
        for params_item in params_list:
            observation = evaluate_configuration(params_item, bitcoin_prices)
            observations.append(observation)

if __name__ == "__main__":
    main()
