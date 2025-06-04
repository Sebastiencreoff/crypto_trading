import logging
import os
import torch
import torch.nn as nn
from crypto_trading.algo.algoIf import AlgoIf

# Define a simple placeholder PyTorch model class
# Input size will be num_price_features + num_indicator_features
# Defaulting to 5 price features + 3 indicator features = 8
class PlaceholderNet(nn.Module):
    def __init__(self, input_size=8, output_size=3):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        # x is expected to be of shape [batch_size, input_size] or [input_size]
        # If it's [input_size], it will be treated as [1, input_size] by nn.Linear
        return self.linear(x)

class AIAlgo(AlgoIf):
    # Define default feature counts at class level for clarity
    NUM_PRICE_FEATURES = 5
    TARGET_INDICATOR_KEYS = ['GuppyMMA', 'Bollinger', 'MovingAverageCrossover']
    # NUM_INDICATOR_FEATURES will be len(TARGET_INDICATOR_KEYS)

    def __init__(self, ai_algo_config): # Parameter name changed
        # super().__init__() # AlgoIf has no __init__, so no need to call super().__init__
        self.model = None
        self.expected_input_size = self.NUM_PRICE_FEATURES + len(self.TARGET_INDICATOR_KEYS)

        # Now ai_algo_config is already the 'AIAlgo' section of the main config
        model_path = ai_algo_config.get("model_path")

        if model_path and os.path.exists(model_path):
            try:
                self.model = torch.load(model_path)
                self.model.eval()
                logging.info(f"Loaded model from {model_path}")
                # Verify loaded model's input size if possible (requires model introspection)
                try:
                    # Attempt to get in_features from the first linear layer found
                    first_linear_layer = next(m for m in self.model.modules() if isinstance(m, nn.Linear))
                    if first_linear_layer.in_features != self.expected_input_size:
                        logging.warning(f"Loaded model expects {first_linear_layer.in_features} features, but AIAlgo is configured for {self.expected_input_size}. Mismatch may cause errors.")
                except StopIteration:
                    logging.warning("Could not find a nn.Linear layer in the loaded model to verify input_size.")
                except Exception as e:
                    logging.warning(f"Could not verify input_size of loaded model: {e}")

            except Exception as e:
                logging.error(f"Error loading model from {model_path}: {e}")
                self.model = PlaceholderNet(input_size=self.expected_input_size)
                logging.warning(f"Using placeholder model (input_size={self.expected_input_size}) due to error loading.")
        else:
            self.model = PlaceholderNet(input_size=self.expected_input_size)
            if model_path:
                logging.warning(f"Model path {model_path} not found. Using placeholder model (input_size={self.expected_input_size}).")
            else:
                logging.warning(f"No model path provided. Using placeholder model (input_size={self.expected_input_size}).")

    def process(self, current_value, historical_values, currency, **kwargs): # Signature updated
        if not self.model:
            logging.error("Model not loaded in AIAlgo.")
            return 0

        indicator_signals = kwargs.get('indicator_signals', {}) # Retrieve from kwargs

        # 1. Extract Indicator Features
        indicator_features = []
        for key in self.TARGET_INDICATOR_KEYS:
            signal = indicator_signals.get(key, 0) # Default to 0 if an indicator signal is missing
            indicator_features.append(float(signal))
            logging.debug(f"AIAlgo: Using signal from {key}: {signal}")

        # 2. Prepare Price Features
        # Use last `NUM_PRICE_FEATURES` from historical_values, or fewer if not available, then pad.
        # The model expects a fixed size, so padding or a different strategy for handling
        # insufficient history is important.

        # We need NUM_PRICE_FEATURES - 1 from historical `historical_values` because current_value is the last one.
        num_historical_needed = self.NUM_PRICE_FEATURES - 1

        # Start with current_value as the most recent price feature
        price_features_prep = [float(current_value)]

        if len(historical_values) >= num_historical_needed: # Changed 'values' to 'historical_values'
            # Sufficient historical data
            price_features_prep = [float(v) for v in historical_values[-num_historical_needed:]] + price_features_prep # Changed 'values'
        else:
            # Insufficient historical data, pad with the oldest available value or current_value if historical_values is empty
            padding_value = float(historical_values[0]) if historical_values else float(current_value) # Changed 'values'
            num_to_pad = num_historical_needed - len(historical_values) # Changed 'values'
            price_features_prep = [padding_value] * num_to_pad + [float(v) for v in historical_values] + price_features_prep # Changed 'values'
            logging.warning(
                f"AIAlgo: Not enough historical price data ({len(historical_values)} points) for {self.NUM_PRICE_FEATURES} features. " # Changed 'values'
                f"Padding with value: {padding_value}."
            )

        # Ensure price_features is exactly NUM_PRICE_FEATURES long (it should be by now)
        price_features = price_features_prep[:self.NUM_PRICE_FEATURES]


        # 3. Combine Features
        combined_features = price_features + indicator_features

        # Verify total number of features
        if len(combined_features) != self.expected_input_size:
            logging.error(
                f"AIAlgo Feature mismatch: Model expected {self.expected_input_size} features, "
                f"but got {len(combined_features)} ({len(price_features)} prices, {len(indicator_features)} indicators). "
                "Returning neutral signal."
            )
            return 0

        logging.debug(f"AIAlgo: Combined features for model: {combined_features}")

        try:
            # Convert data to tensor. Model expects [features] or [batch_size, features]
            # PlaceholderNet is simple nn.Linear, so a 1D tensor of features is fine for a single instance.
            input_tensor = torch.tensor(combined_features, dtype=torch.float32)
            # If model strictly requires batch_size dimension: input_tensor = input_tensor.unsqueeze(0)

            # Inference
            with torch.no_grad():
                output = self.model(input_tensor)

            # Output Conversion
            # Output is a tensor of 3 values (sell, hold, buy probabilities/logits)
            # Example: argmax to get the index of the highest value
            decision_index = torch.argmax(output).item()

            # Map index to signal: 0 -> sell (-1), 1 -> hold (0), 2 -> buy (1)
            if decision_index == 0:
                return -1 # Sell
            elif decision_index == 1:
                return 0  # Hold
            elif decision_index == 2:
                return 1  # Buy
            else:
                logging.warning(f"Unexpected decision index: {decision_index}")
                return 0 # Neutral on unexpected index
        except Exception as e:
            logging.error(f"Error during AIAlgo process: {e}")
            return 0 # Neutral signal on error
