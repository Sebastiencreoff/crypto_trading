import streamlit as st
import json
import os

# Configuration directory
CONFIG_DIR = "config"

# Helper function to load JSON configuration files
def load_config(filename):
    filepath = os.path.join(CONFIG_DIR, filename)
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error: Configuration file '{filename}' not found at '{filepath}'. Returning empty config.")
        return {}
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from '{filename}'. Returning empty config.")
        return {}

# Helper function to save JSON configuration files
def save_config(filename, data):
    filepath = os.path.join(CONFIG_DIR, filename)
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        # st.sidebar.success(f"Saved '{filename}' successfully!") # Minor feedback, consider main area if more prominent
    except Exception as e:
        st.error(f"Error saving configuration file '{filename}': {e}")


# Initial loading of configurations into session_state
if 'algo_config_orig' not in st.session_state:
    st.session_state.algo_config_orig = load_config("algo.json")
if 'connect_simu_config_orig' not in st.session_state:
    st.session_state.connect_simu_config_orig = load_config("connectSimu.json")
if 'trading_coinbase_config_orig' not in st.session_state:
    st.session_state.trading_coinbase_config_orig = load_config("trading_COINBASE.json")
if 'trading_simu_config_orig' not in st.session_state:
    st.session_state.trading_simu_config_orig = load_config("trading_SIMU.json")


st.title("Crypto Trading Bot Configuration")

# Trading Mode Selection
# Ensure the key for the selectbox is consistently used for accessing its value
st.session_state.selected_trading_mode = st.selectbox(
    "Trading Mode",
    ["Coinbase", "Simulation"],
    index=["Coinbase", "Simulation"].index(st.session_state.get('selected_trading_mode', "Coinbase")), # Preserve selection across reruns
    key="trading_mode_selector_widget_key" # Use a distinct key for the widget itself
)
mode = st.session_state.selected_trading_mode # Use the value from session_state

if mode == "Coinbase":
    st.subheader("Coinbase Configuration")
    st.caption(f"Connection Type: {st.session_state.trading_coinbase_config_orig.get('connection', {}).get('type', 'N/A')}")
    st.text_input("API Key (Placeholder - Not Saved)", key="coinbase_api_key_placeholder", type="password", help="API keys are typically not configured via UI for security reasons.")
    st.text_input("API Secret (Placeholder - Not Saved)", key="coinbase_api_secret_placeholder", type="password")

    st.subheader("Coinbase Trading Parameters")
    st.session_state.db_file_coinbase_input = st.text_input(
        "Database File (Coinbase)",
        value=st.session_state.trading_coinbase_config_orig.get("database_file", "COINBASE"),
        key="db_file_coinbase_widget_key"
    )
    st.session_state.currency_coinbase_input = st.text_input(
        "Currency (Coinbase)",
        value=st.session_state.trading_coinbase_config_orig.get("currency", "BTC-USD"),
        key="currency_coinbase_widget_key"
    )
    st.session_state.tx_amt_coinbase_input = st.number_input(
        "Transaction Amount (Coinbase)",
        value=float(st.session_state.trading_coinbase_config_orig.get("transactionAmt", 10.0)),
        min_value=0.0,
        step=0.01,
        key="tx_amt_coinbase_widget_key"
    )
    st.session_state.delay_coinbase_input = st.number_input(
        "Delay (seconds) (Coinbase)",
        value=int(st.session_state.trading_coinbase_config_orig.get("delay", 300)),
        min_value=1,
        step=1,
        key="delay_coinbase_widget_key"
    )

elif mode == "Simulation":
    st.subheader("Simulation Connection Parameters")
    st.caption(f"Connection Type: {st.session_state.connect_simu_config_orig.get('type', 'file')}")
    st.session_state.sim_file_path_input = st.text_input(
        "Simulation File Path",
        value=st.session_state.connect_simu_config_orig.get("file", "inputs/BTC_2019-04-18_to_2019-04-24.csv"),
        key="sim_file_path_widget_key"
    )
    st.session_state.sim_fee_percent_input = st.number_input(
        "Fee Percent (Simulation)",
        value=float(st.session_state.connect_simu_config_orig.get("fee_percent", 0.005)),
        min_value=0.0,
        max_value=1.0,
        step=0.0001,
        format="%.4f",
        key="sim_fee_percent_widget_key"
    )

    st.subheader("Simulation Trading Parameters")
    st.caption(f"Connection Type: {st.session_state.trading_simu_config_orig.get('connection', {}).get('type', 'simulation')}")
    st.session_state.db_file_simu_input = st.text_input(
        "Database File (Simulation)",
        value=st.session_state.trading_simu_config_orig.get("database_file", "SIMU"),
        key="db_file_simu_widget_key"
    )
    st.session_state.currency_simu_input = st.text_input(
        "Currency (Simulation)",
        value=st.session_state.trading_simu_config_orig.get("currency", "BTC-USD"),
        key="currency_simu_widget_key"
    )
    st.session_state.tx_amt_simu_input = st.number_input(
        "Transaction Amount (Simulation)",
        value=float(st.session_state.trading_simu_config_orig.get("transactionAmt", 1000.0)),
        min_value=0.0,
        step=0.01,
        key="tx_amt_simu_widget_key"
    )
    st.session_state.delay_simu_input = st.number_input(
        "Delay (seconds) (Simulation)",
        value=int(st.session_state.trading_simu_config_orig.get("delay", 300)),
        min_value=1,
        step=1,
        key="delay_simu_widget_key"
    )


st.subheader("Algorithm Configuration (Shared for all modes)")

algo_config_orig_loaded = st.session_state.algo_config_orig # Use a shorter alias

# Guppy MMA
guppy_mma_defaults = algo_config_orig_loaded.get("GuppyMMA", {})
st.markdown("#### Guppy Multiple Moving Average (GuppyMMA)")
st.session_state.guppy_short_term_str_input = st.text_input(
    "Short Term Periods (comma-separated integers)",
    value=", ".join(map(str, guppy_mma_defaults.get("short_term", [3, 5, 8, 10, 12, 15]))),
    key="guppy_short_term_widget_key"
)
st.session_state.guppy_long_term_str_input = st.text_input(
    "Long Term Periods (comma-separated integers)",
    value=", ".join(map(str, guppy_mma_defaults.get("long_term", [30, 35, 40, 45, 50, 60]))),
    key="guppy_long_term_widget_key"
)
st.session_state.guppy_buy_input = st.number_input(
    "Buy Threshold (GuppyMMA)",
    value=float(guppy_mma_defaults.get("buy", 0.0)),
    step=0.01,
    key="guppy_buy_widget_key"
)
st.session_state.guppy_sell_input = st.number_input(
    "Sell Threshold (GuppyMMA)",
    value=float(guppy_mma_defaults.get("sell", 0.0)),
    step=0.01,
    key="guppy_sell_widget_key"
)

# Max Lost
max_lost_defaults = algo_config_orig_loaded.get("maxLost", {})
st.markdown("#### Max Lost")
st.session_state.max_lost_percentage_input = st.number_input(
    "Percentage (Max Lost)",
    value=float(max_lost_defaults.get("percentage", 0.10)),
    min_value=0.0,
    max_value=1.0,
    step=0.001,
    format="%.3f",
    key="max_lost_percentage_widget_key",
    help="Enter as a decimal (e.g., 0.1 for 10%)"
)

# Take Profit
take_profit_defaults = algo_config_orig_loaded.get("takeProfit", {})
st.markdown("#### Take Profit")
st.session_state.take_profit_percentage_input = st.number_input(
    "Percentage (Take Profit)",
    value=float(take_profit_defaults.get("percentage", 0.05)),
    min_value=0.0,
    step=0.0001,
    format="%.4f",
    key="take_profit_percentage_widget_key",
    help="Enter as a decimal (e.g., 0.05 for 5%)"
)

# Bollinger Bands
bollinger_defaults = algo_config_orig_loaded.get("Bollinger", {})
st.markdown("#### Bollinger Bands")
st.session_state.bollinger_frequency_input = st.number_input(
    "Frequency (Bollinger)",
    value=int(bollinger_defaults.get("frequency", 20)),
    min_value=1,
    step=1,
    key="bollinger_frequency_widget_key"
)
st.session_state.bollinger_std_dev_input = st.number_input(
    "Standard Deviations (Bollinger)",
    value=float(bollinger_defaults.get("std_dev", 2.0)),
    min_value=0.1,
    step=0.1,
    key="bollinger_std_dev_widget_key"
)

if st.button("Save Configuration", key="save_button_widget_key"):
    errors_found = False
    # --- Save Algorithm Configuration (algo.json) ---
    new_algo_config = {}
    # GuppyMMA
    try:
        short_term_periods = [int(x.strip()) for x in st.session_state.guppy_short_term_str_input.split(',') if x.strip()]
        long_term_periods = [int(x.strip()) for x in st.session_state.guppy_long_term_str_input.split(',') if x.strip()]
        new_algo_config["GuppyMMA"] = {
            "short_term": short_term_periods,
            "long_term": long_term_periods,
            "buy": st.session_state.guppy_buy_input,
            "sell": st.session_state.guppy_sell_input,
        }
    except ValueError:
        st.error("Error: Invalid format for GuppyMMA short/long term periods. Please use comma-separated integers.")
        errors_found = True

    # Max Lost - preserve other keys
    original_max_lost = st.session_state.algo_config_orig.get("maxLost", {})
    new_algo_config["maxLost"] = {**original_max_lost, "percentage": st.session_state.max_lost_percentage_input}

    # Take Profit - preserve other keys (if any, though current structure only has percentage)
    original_take_profit = st.session_state.algo_config_orig.get("takeProfit", {})
    new_algo_config["takeProfit"] = {**original_take_profit, "percentage": st.session_state.take_profit_percentage_input}

    # Bollinger - preserve other keys
    original_bollinger = st.session_state.algo_config_orig.get("Bollinger", {})
    new_algo_config["Bollinger"] = {
        **original_bollinger, # Preserve any other sub-keys
        "frequency": st.session_state.bollinger_frequency_input,
        "std_dev": st.session_state.bollinger_std_dev_input # Assuming std_dev was added as editable
    }
    # Preserve any other top-level keys from original algo.json not handled by UI
    for key, value in st.session_state.algo_config_orig.items():
        if key not in new_algo_config:
            new_algo_config[key] = value

    if not errors_found:
        save_config("algo.json", new_algo_config)

    # --- Save Simulation Connection Configuration (connectSimu.json) ---
    # This is saved regardless of mode, as it's independent
    new_connect_simu_config = {
        "type": st.session_state.connect_simu_config_orig.get("type", "file"), # Preserve original type
        "file": st.session_state.sim_file_path_input,
        "fee_percent": st.session_state.sim_fee_percent_input
    }
    # Preserve any other top-level keys
    for key, value in st.session_state.connect_simu_config_orig.items():
        if key not in new_connect_simu_config:
            new_connect_simu_config[key] = value
    save_config("connectSimu.json", new_connect_simu_config)


    # --- Save Trading Configuration (Conditional on mode) ---
    current_mode_selected = st.session_state.selected_trading_mode # Use the state variable
    if current_mode_selected == "Coinbase":
        new_trading_coinbase_config = {
            "database_file": st.session_state.db_file_coinbase_input,
            "currency": st.session_state.currency_coinbase_input,
            "transactionAmt": st.session_state.tx_amt_coinbase_input,
            "delay": st.session_state.delay_coinbase_input,
            # Preserve non-editable fields
            "connection": st.session_state.trading_coinbase_config_orig.get("connection"),
            "connectionConfig": st.session_state.trading_coinbase_config_orig.get("connectionConfig"),
            "algoConfig": st.session_state.trading_coinbase_config_orig.get("algoConfig")
        }
        # Preserve any other top-level keys
        for key, value in st.session_state.trading_coinbase_config_orig.items():
            if key not in new_trading_coinbase_config:
                new_trading_coinbase_config[key] = value
        save_config("trading_COINBASE.json", new_trading_coinbase_config)
    elif current_mode_selected == "Simulation":
        new_trading_simu_config = {
            "database_file": st.session_state.db_file_simu_input,
            "currency": st.session_state.currency_simu_input,
            "transactionAmt": st.session_state.tx_amt_simu_input,
            "delay": st.session_state.delay_simu_input,
            # Preserve non-editable fields
            "connection": st.session_state.trading_simu_config_orig.get("connection"),
            "connectionConfig": st.session_state.trading_simu_config_orig.get("connectionConfig"),
            "algoConfig": st.session_state.trading_simu_config_orig.get("algoConfig")
        }
        # Preserve any other top-level keys
        for key, value in st.session_state.trading_simu_config_orig.items():
            if key not in new_trading_simu_config:
                new_trading_simu_config[key] = value
        save_config("trading_SIMU.json", new_trading_simu_config)

    if not errors_found:
        st.success("Configuration saved successfully!")
        # Reload configurations to reflect saved state as new "original"
        st.session_state.algo_config_orig = load_config("algo.json")
        st.session_state.connect_simu_config_orig = load_config("connectSimu.json")
        st.session_state.trading_coinbase_config_orig = load_config("trading_COINBASE.json")
        st.session_state.trading_simu_config_orig = load_config("trading_SIMU.json")
        # Force rerun to update UI with reloaded values if necessary, though Streamlit should do it.
        # st.experimental_rerun() # Use with caution, usually not needed if session_state is managed correctly.
    else:
        st.error("Some configurations were not saved due to errors. Please check messages above.")
