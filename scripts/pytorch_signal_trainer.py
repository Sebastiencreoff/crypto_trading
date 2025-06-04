#! /usr/bin/env python

import requests
import time
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import sys # For path modification
import os # For path joining
import json # Added for robust error handling in download_bitcoin_data

# Add parent dir to path to allow imports from crypto_trading if needed later
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def download_bitcoin_data(days=365):
    """
    Downloads the last `days` of Bitcoin (BTC) daily price data vs USD from CoinGecko.
    Returns a list of prices.
    """
    coin_id = "bitcoin"
    vs_currency = "usd"
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"

    params = {
        "vs_currency": vs_currency,
        "days": str(days),
        "interval": "daily"
    }

    prices_list = []
    try:
        print(f"Downloading data for {days} days...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        prices_list = [item[1] for item in data.get("prices", [])]

        if not prices_list:
            print("Warning: Downloaded price list is empty.")
            return []

        if len(prices_list) > days:
             prices_list = prices_list[-(days+1):-1] if len(prices_list) == days + 1 else prices_list[-days:]


    except requests.exceptions.RequestException as e:
        print(f"Error downloading Bitcoin data: {e}")
        return []
    except json.JSONDecodeError:
        print("Error decoding JSON response from CoinGecko.")
        return []
    except KeyError:
        print("Error: 'prices' key not found in CoinGecko response.")
        return []

    return prices_list

def create_sequences_and_labels(prices_list, sequence_length, look_forward_days, price_change_threshold_percent):
    """
    Creates sequences, labels, and returns the scaler.
    Note: In a real scenario, scaler should be fit on training data only.
          This function currently fits on all provided `prices_list`.
    """
    if not prices_list or len(prices_list) < sequence_length + look_forward_days:
        print("Error: Not enough data to create sequences and labels.")
        return np.array([]), np.array([]), None

    prices_np = np.array(prices_list).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices_np)

    sequences = []
    labels = []

    max_start_idx = len(scaled_prices) - sequence_length - look_forward_days

    for i in range(max_start_idx + 1):
        seq_end_idx = i + sequence_length
        current_price_for_label = prices_list[seq_end_idx - 1]
        future_price_for_label = prices_list[seq_end_idx -1 + look_forward_days]

        label = 1 # HOLD
        if future_price_for_label >= current_price_for_label * (1 + price_change_threshold_percent / 100):
            label = 2 # BUY
        elif future_price_for_label <= current_price_for_label * (1 - price_change_threshold_percent / 100):
            label = 0 # SELL

        sequence = scaled_prices[i:seq_end_idx].flatten()
        sequences.append(sequence)
        labels.append(label)

    if not sequences:
        print("Warning: No sequences were created. Check input data length and parameters.")
        return np.array([]), np.array([]), scaler

    return np.array(sequences), np.array(labels), scaler


class BitcoinSignalDataset(Dataset):
    """PyTorch Dataset for Bitcoin trading signals."""
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sequence.unsqueeze(1), label

class SignalLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=3):
        super(SignalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

def predict_signal(model, raw_price_sequence, device, sequence_length, scaler):
    """
    Predicts a trading signal (BUY, SELL, HOLD) for a given raw price sequence.
    """
    # try:
    #     if len(raw_price_sequence) != sequence_length:
    #         print(f"Error: Input sequence length must be {sequence_length}. Got {len(raw_price_sequence)}")
    #         return None
    #     model.eval()
    #     price_sequence_np = np.array(raw_price_sequence).reshape(-1, 1)
    #     scaled_sequence = scaler.transform(price_sequence_np)
    #     input_tensor = torch.tensor(scaled_sequence.flatten(), dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(device)
    #     with torch.no_grad():
    #         output = model(input_tensor)
    #         _, predicted_idx = torch.max(output, 1)
    #         return predicted_idx.item()
    # except Exception as e:
    #     print(f"Error during prediction: {e}")
    #     return None
    pass

if __name__ == "__main__":
    print("Starting PyTorch Signal Trainer script...")

    # --- Configuration Parameters ---
    DAYS_TO_DOWNLOAD = 365 * 2 # Download 2 years of data
    SEQUENCE_LENGTH = 30       # Use last 30 days of prices to predict
    LOOK_FORWARD_DAYS = 7      # Predict signal based on price change over next 7 days
    PRICE_CHANGE_THRESHOLD = 2.0 # Percentage change threshold for BUY/SELL signal

    TRAIN_SPLIT_RATIO = 0.8
    BATCH_SIZE = 32 # Adjusted from min(4, len) for a more typical default

    # # --- PyTorch Model and Training Hyperparameters (for when environment supports it) ---
    # # LSTM Model parameters
    # INPUT_SIZE = 1 # Number of features (just price for now)
    # HIDDEN_SIZE = 64
    # NUM_LAYERS = 2
    # OUTPUT_SIZE = 3 # BUY (2), HOLD (1), SELL (0)
    #
    # # Training hyperparameters
    # LEARNING_RATE = 0.001
    # EPOCHS = 20 # Number of epochs to train for
    # # Define device
    # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # print(f"Using device: {device}")
    # # --------------------------------------------------------------------------

    scaler_instance = None

    # --- [Step 1] Data Download ---
    print(f"\n[Step 1] Downloading Bitcoin data for {DAYS_TO_DOWNLOAD} days...")
    raw_prices = download_bitcoin_data(days=DAYS_TO_DOWNLOAD)

    if raw_prices and len(raw_prices) >= SEQUENCE_LENGTH + LOOK_FORWARD_DAYS : # Ensure enough data for at least one sequence
        print(f"Successfully downloaded {len(raw_prices)} price points.")

        # --- [Step 2] Preprocessing: Create Sequences and Labels ---
        print(f"\n[Step 2] Creating sequences and labels...")
        all_sequences, all_labels, scaler_instance = create_sequences_and_labels(
            raw_prices,
            SEQUENCE_LENGTH,
            LOOK_FORWARD_DAYS,
            PRICE_CHANGE_THRESHOLD
        )

        if all_sequences.size > 0 and all_labels.size > 0:
            print(f"Total sequences created: {all_sequences.shape[0]}")
            print(f"Total labels created: {all_labels.shape[0]}")
            if scaler_instance:
                print("Scaler was fitted during preprocessing.")

            # --- [Step 3] Data Splitting ---
            print(f"\n[Step 3] Splitting data into training and validation sets...")
            split_idx = int(len(all_sequences) * TRAIN_SPLIT_RATIO)
            train_sequences = all_sequences[:split_idx]
            train_labels = all_labels[:split_idx]
            val_sequences = all_sequences[split_idx:]
            val_labels = all_labels[split_idx:]
            print(f"Training set: {len(train_sequences)} sequences")
            print(f"Validation set: {len(val_sequences)} sequences")

            # --- [Step 4] Create Datasets and DataLoaders ---
            print(f"\n[Step 4] Creating PyTorch Datasets and DataLoaders...")
            train_dataset = BitcoinSignalDataset(train_sequences, train_labels)
            val_dataset = BitcoinSignalDataset(val_sequences, val_labels) # Placeholder for val_dataset

            if len(train_dataset) > 0:
                train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
                print(f"Train DataLoader created with batch size: {BATCH_SIZE}")
                 # # val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) # Commented for now
                 # # print(f"Validation DataLoader created with batch size: {BATCH_SIZE}")
            else:
                print("Training dataset is empty. Cannot create DataLoader.")
                train_dataloader = None # Explicitly set to None
                # val_dataloader = None

            # --- [Step 5] DataLoader Verification (if train_dataloader exists) ---
            if train_dataloader:
                print(f"\n[Step 5] Verifying Train DataLoader - fetching one batch...")
                try:
                    sample_sequences_batch, sample_labels_batch = next(iter(train_dataloader))
                    print(f"Sample train sequences batch shape: {sample_sequences_batch.shape}")
                    print(f"Sample train labels batch shape: {sample_labels_batch.shape}")
                except StopIteration:
                    print("Could not get a batch from Train DataLoader.")

            # --- [Step 6] Model Definition & Setup (Commented Out) ---
            # # print("\n[Step 6] Model Definition & Setup (Commented Out)")
            # # INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE defined in config section
            # # model = SignalLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
            # # model.to(device)
            # # criterion = nn.CrossEntropyLoss()
            # # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            # # print("Model, Loss, Optimizer defined (not executed due to env limits).")
            # # -----------------------------------------------------------

            # --- [Step 7] Training Loop (Commented Out) ---
            # # print("\n[Step 7] Training Loop (Commented Out)")
            # # best_accuracy = 0.0 # Initialize best_accuracy outside the loop
            # # for epoch in range(EPOCHS):
            # #     model.train()
            # #     running_loss = 0.0
            # #     for batch_idx, (sequences, labels) in enumerate(train_dataloader):
            # #         sequences, labels = sequences.to(device), labels.to(device)
            # #         optimizer.zero_grad()
            # #         outputs = model(sequences)
            # #         loss = criterion(outputs, labels)
            # #         loss.backward()
            # #         optimizer.step()
            # #         running_loss += loss.item()
            # #         if (batch_idx + 1) % 10 == 0:
            # #             print(f'Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx+1}/{len(train_dataloader)}], Train Loss: {loss.item():.4f}')
            # #     print(f'Epoch [{epoch+1}/{EPOCHS}] completed. Average Training Loss: {running_loss / len(train_dataloader):.4f}')
            # #
            # #     # --- Validation Loop (Commented Out) ---
            # #     # if val_dataloader:
            # #     #     model.eval()
            # #     #     val_loss = 0.0
            # #     #     correct_predictions = 0
            # #     #     total_predictions = 0
            # #     #     with torch.no_grad():
            # #     #         for sequences, labels in val_dataloader:
            # #     #             sequences, labels = sequences.to(device), labels.to(device)
            # #     #             outputs = model(sequences)
            # #     #             loss = criterion(outputs, labels)
            # #     #             val_loss += loss.item()
            # #     #             _, predicted_labels = torch.max(outputs, 1)
            # #     #             total_predictions += labels.size(0)
            # #     #             correct_predictions += (predicted_labels == labels).sum().item()
            # #     #     avg_val_loss = val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0
            # #     #     accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
            # #     #     print(f'Epoch [{epoch+1}/{EPOCHS}], Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')
            # #     #     if accuracy > best_accuracy:
            # #     #         best_accuracy = accuracy
            # #     #         # torch.save(model.state_dict(), "signal_lstm_model.pth")
            # #     #         # print(f"Model saved with accuracy: {accuracy:.2f}%")
            # # ----------------------------------------------------------------

            # --- [Step 8] Prediction Example (Commented Out) ---
            # # print("\n[Step 8] Prediction Example (Commented Out)")
            # # # This assumes 'model' would be loaded or is the trained model from above,
            # # # 'device' is defined, and 'scaler_instance' is the one fitted on training data.
            # # if scaler_instance and raw_prices and len(raw_prices) >= SEQUENCE_LENGTH:
            # #     # For a real prediction, you'd likely load a saved model:
            # #     # model_for_prediction = SignalLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
            # #     # model_for_prediction.load_state_dict(torch.load("signal_lstm_model.pth"))
            # #     # model_for_prediction.to(device)
            # #     # model_for_prediction.eval() # Ensure it's in eval mode
            # #
            # #     # Example: use the last sequence_length of available raw_prices
            # #     example_raw_prices = raw_prices[-SEQUENCE_LENGTH:]
            # #     print(f"Using last {SEQUENCE_LENGTH} raw prices for prediction: {example_raw_prices[:5]}...") # Print first 5
            # #
            # #     # predicted_signal_idx = predict_signal(model_for_prediction, example_raw_prices, device, SEQUENCE_LENGTH, scaler_instance)
            # #     predicted_signal_idx = 1 # Placeholder, as actual call is commented
            # #
            # #     if predicted_signal_idx is not None:
            # #         signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
            # #         print(f"Predicted signal: {signal_map.get(predicted_signal_idx, 'Unknown')}")
            # #     else:
            # #         print("Prediction could not be made.")
            # # else:
            # #    print("Skipping prediction example: Model, scaler, or sufficient price data not available.")
            # # ---------------------------------------------------------------------
        else:
            print("Failed to create sequences or labels; arrays are empty. Cannot proceed with data splitting or DataLoader creation.")
    else:
        print("Failed to download or process Bitcoin data. Price list is empty or too short for sequence creation.")

    print("\nPyTorch Signal Trainer script finished.")
