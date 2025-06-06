import logging
import os
import torch
import torch.nn as nn
from crypto_trading.algo.algoIf import AlgoIf

# Define a simple placeholder PyTorch model class
# Input size will be num_price_features + num_indicator_features
# Defaulting to 5 price features + 3 indicator features = 8
# Output vector mapping (14 outputs):
# Indices 0-2: AIAlgo's own signal decision (logits for sell, hold, buy)
# Index 3: GuppyMMA - num_short_terms (e.g., scaled to produce 3-6 terms)
# Index 4: GuppyMMA - short_term_start (e.g., scaled to produce 3-10)
# Index 5: GuppyMMA - short_term_step (e.g., scaled to produce 1-3)
# Index 6: GuppyMMA - num_long_terms (e.g., scaled to produce 3-6 terms)
# Index 7: GuppyMMA - long_term_start (e.g., scaled to produce 30-50)
# Index 8: GuppyMMA - long_term_step (e.g., scaled to produce 5-10)
# Index 9: GuppyMMA - buy_threshold_factor (scaled to 0.0-1.0)
# Index 10: GuppyMMA - sell_threshold_factor (scaled to 0.0-1.0)
# Index 11: Bollinger - frequency (e.g., scaled to produce 10-50)
# Index 12: MovingAverageCrossover - short_window (e.g., scaled to produce 5-50)
# Index 13: MovingAverageCrossover - long_window_offset (e.g., scaled to produce 5-50, added to short_window)
class PlaceholderNet(nn.Module):
    def __init__(self, input_size=8, output_size=14): # Updated output_size
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

class AIAlgo(AlgoIf):
    NUM_PRICE_FEATURES = 5
    TARGET_INDICATOR_KEYS = ['GuppyMMA', 'Bollinger', 'MovingAverageCrossover']
    # For PlaceholderNet, the output size for parameters should be 11 (14 total - 3 for AI signal)
    MODEL_OUTPUT_SIZE = 14 # 3 for AI signal + 11 for parameters

    def __init__(self, config_dict):
        super().__init__(config_dict)
        # self.algo_configs is used by get_target_algo_configs for initial setup.
        # It's not directly used by AIAlgo.process anymore for dynamic updates.
        self.algo_configs = {
            'GuppyMMA': {
                "short_term": [3, 5, 8, 10, 12, 15], "long_term": [30, 35, 40, 45, 50, 60],
                "buy": 6, "sell": 6
            },
            'Bollinger': {"frequency": 20}, # Default from original Bollinger, example
            'MovingAverageCrossover': {"short_window": 20, "long_window": 50}
        }

        self.model = None
        self.expected_input_size = self.NUM_PRICE_FEATURES + len(self.TARGET_INDICATOR_KEYS)
        model_path = config_dict.get("AIAlgo", {}).get("model_path")

        if model_path and os.path.exists(model_path):
            try:
                self.model = torch.load(model_path)
                self.model.eval()
                logging.info(f"Loaded model from {model_path}")
                # Model introspection for input/output size (example)
                first_linear_layer = next((m for m in self.model.modules() if isinstance(m, nn.Linear)), None)
                if first_linear_layer:
                    if first_linear_layer.in_features != self.expected_input_size:
                        logging.warning(f"Loaded model expects input_size={first_linear_layer.in_features}, AIAlgo configured for {self.expected_input_size}.")
                    if first_linear_layer.out_features != self.MODEL_OUTPUT_SIZE:
                        logging.warning(f"Loaded model expects output_size={first_linear_layer.out_features}, AIAlgo configured for {self.MODEL_OUTPUT_SIZE}.")
                else:
                    logging.warning("Could not find a nn.Linear layer in loaded model to verify input/output sizes.")
            except Exception as e:
                logging.error(f"Error loading model from {model_path}: {e}")
                self.model = PlaceholderNet(input_size=self.expected_input_size, output_size=self.MODEL_OUTPUT_SIZE)
                logging.warning(f"Using placeholder model (input_size={self.expected_input_size}, output_size={self.MODEL_OUTPUT_SIZE}) due to error loading.")
        else:
            self.model = PlaceholderNet(input_size=self.expected_input_size, output_size=self.MODEL_OUTPUT_SIZE)
            logging.warning(f"Model path '{model_path}' not found or not provided. Using placeholder model (input_size={self.expected_input_size}, output_size={self.MODEL_OUTPUT_SIZE}).")

    def process(self, current_value, values, currency, indicator_signals):
        if not self.model:
            logging.error("Model not loaded in AIAlgo.")
            return 0, {} # Neutral signal, empty configs

        # 1. Extract Indicator Features
        indicator_features = [float(indicator_signals.get(key, 0)) for key in self.TARGET_INDICATOR_KEYS]

        # 2. Prepare Price Features (Simplified for brevity, original logic kept if suitable)
        num_historical_needed = self.NUM_PRICE_FEATURES - 1
        price_features_prep = [float(current_value)]
        if len(values) >= num_historical_needed:
            price_features_prep = [float(v) for v in values[-num_historical_needed:]] + price_features_prep
        else:
            padding_value = float(values[0]) if values else float(current_value)
            num_to_pad = num_historical_needed - len(values)
            price_features_prep = [padding_value] * num_to_pad + [float(v) for v in values] + price_features_prep
        price_features = price_features_prep[:self.NUM_PRICE_FEATURES]

        # 3. Combine Features
        combined_features = price_features + indicator_features
        if len(combined_features) != self.expected_input_size:
            logging.error(f"AIAlgo Feature mismatch: Model expected {self.expected_input_size}, got {len(combined_features)}. Returning neutral signal.")
            return 0, {}

        logging.debug(f"AIAlgo: Combined features for model: {combined_features}")

        try:
            input_tensor = torch.tensor(combined_features, dtype=torch.float32)
            with torch.no_grad():
                output = self.model(input_tensor) # Output is now 14 values

            # AI's own signal decision (from first 3 outputs)
            decision_index = torch.argmax(output[:3]).item()
            ai_own_signal = 0
            if decision_index == 0: ai_own_signal = -1 # Sell
            elif decision_index == 2: ai_own_signal = 1  # Buy

            # Parse and prepare configurations for other algos
            algo_configs_map = {}

            # GuppyMMA (indices 3-10)
            raw_num_short = output[3].item()
            raw_start_short = output[4].item()
            raw_step_short = output[5].item()
            raw_num_long = output[6].item()
            raw_start_long = output[7].item()
            raw_step_long = output[8].item()
            buy_factor_tensor = output[9] # Keep as tensor for sigmoid
            sell_factor_tensor = output[10] # Keep as tensor for sigmoid

            num_short_terms = max(3, min(6, int(raw_num_short)))
            short_term_start = max(3, min(10, int(raw_start_short)))
            short_term_step = max(1, min(3, int(raw_step_short)))
            num_long_terms = max(3, min(6, int(raw_num_long)))
            long_term_start = max(30, min(50, int(raw_start_long)))
            long_term_step = max(5, min(10, int(raw_step_long)))

            buy_factor = torch.sigmoid(buy_factor_tensor).item()
            sell_factor = torch.sigmoid(sell_factor_tensor).item()

            short_terms_list = [short_term_start + i * short_term_step for i in range(num_short_terms)]
            long_terms_list = [long_term_start + i * long_term_step for i in range(num_long_terms)]

            buy_threshold = max(1, int(buy_factor * num_short_terms)) if num_short_terms > 0 else 0
            sell_threshold = max(1, int(sell_factor * num_short_terms)) if num_short_terms > 0 else 0

            algo_configs_map['GuppyMMA'] = {
                'short_term': short_terms_list,
                'long_term': long_terms_list,
                'buy': buy_threshold,
                'sell': sell_threshold
            }

            # Bollinger (index 11)
            raw_bollinger_freq = output[11].item()
            bollinger_frequency = max(5, min(100, int(raw_bollinger_freq))) # Example: 5-100
            algo_configs_map['Bollinger'] = {'frequency': bollinger_frequency}

            # MovingAverageCrossover (indices 12-13)
            raw_mac_short = output[12].item()
            raw_mac_offset = output[13].item()
            mac_short_window = max(5, min(100, int(raw_mac_short))) # Example: 5-100
            mac_long_offset = max(5, min(100, int(raw_mac_offset))) # Example: 5-100, offset from short
            mac_long_window = mac_short_window + mac_long_offset
            algo_configs_map['MovingAverageCrossover'] = {
                'short_window': mac_short_window,
                'long_window': mac_long_window
            }

            logging.debug(f"AIAlgo generated new algo_configs: {algo_configs_map}")
            return ai_own_signal, algo_configs_map

        except Exception as e:
            logging.error(f"Error during AIAlgo process: {e}", exc_info=True)
            return 0, {} # Neutral signal, empty configs on error

    def get_target_algo_configs(self):
        """
        Returns the initial default configurations for other algorithms.
        This is used by AlgoMain to set up the algos before AIAlgo starts providing dynamic updates.
        """
        return self.algo_configs
