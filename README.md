# pythonTrading
Automated cryptocurrency trading bot.

This project provides a framework for implementing and running various trading algorithms for cryptocurrencies.

## Features
- Multiple algorithm support (GuppyMMA, Bollinger Bands, Moving Average Crossover, AI-based).
- Configuration via JSON files.
- Extensible algorithm interface (`AlgoIf`).

## Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt` (Note: `requirements.txt` may need to be created or updated).
3. Configure your trading parameters and API keys in the relevant configuration files.

## Exchange Configuration

This section details how to configure the bot to connect to different cryptocurrency exchanges. Currently, Binance is the primary supported exchange for live/API-based trading. Simulation mode is also available.

### Binance
Binance is a supported exchange for trading. Configuration involves two main files:

1.  **`config/trading_BINANCE.json`**: This file defines the trading strategy parameters for Binance, such as the currency pair (e.g., `BTCUSDT`), transaction amounts, and which algorithm configurations to use.
2.  **`config/binance.json`**: This file stores your Binance API credentials and a simulation flag.

**Setup Instructions for Binance:**

1.  **Create API Key File**:
    You will need to create a `config/binance.json` file. You can do this by copying the example template:
    ```bash
    cp config/testing_binance_api.json config/binance.json
    ```
    Then, edit `config/binance.json`.

2.  **Add Your API Credentials**:
    In your newly created `config/binance.json`, replace `"YOUR_BINANCE_API_KEY"` and `"YOUR_BINANCE_API_SECRET"` with your actual Binance API key and secret.

    ```json
    {
        "api_key": "YOUR_ACTUAL_BINANCE_API_KEY",
        "api_secret": "YOUR_ACTUAL_BINANCE_API_SECRET",
        "simulation": true
    }
    ```

3.  **Verify Trading Configuration**:
    Ensure the `connectionConfig` path in `config/trading_BINANCE.json` correctly points to your API key file. The default is `"config/binance.json"`, so if you named your API key file `binance.json` and placed it in the `config/` directory, this should already be correct.
    ```json
    // In config/trading_BINANCE.json
    {
        // ... other settings ...
        "connection": "binance",
        "connectionConfig": "config/binance.json", // Make sure this points to your API key file
        // ... other settings ...
    }
    ```

4.  **Simulation Mode**:
    -   Set `"simulation": false` in `config/binance.json` for live trading with real funds.
    -   Set `"simulation": true` for simulated trading. The current `BinanceConnect` implementation uses the live Binance API endpoints. If `simulation` is true, it doesn't explicitly use Binance's testnet. The `python-binance` library itself might support testnet environments if API keys for a testnet account are provided and the library is configured accordingly, but `BinanceConnect` does not manage this switch automatically. For true paper trading without hitting live endpoints with dummy orders, ensure the library is configured for testnet or use a dedicated simulation mode if the connector supports it. *Developer Note: The current `BinanceConnect` does not have explicit testnet endpoint switching logic; this relies on how `python-binance` handles keys that might be for a testnet account.*

## Algorithms

### Guppy Multiple Moving Average (GuppyMMA)
- Description of GuppyMMA and its configuration.

### Bollinger Bands
- Description of Bollinger Bands and its configuration.

### Moving Average Crossover
- Description of Moving Average Crossover and its configuration.

### AI Algorithm (`AIAlgo`)
The `AIAlgo` is a trading algorithm module that utilizes a PyTorch-based neural network to generate trading signals (buy, sell, hold). It implements the `AlgoIf` interface and is managed by the main algorithm processor (`AlgoMain`).

**PyTorch Dependency:**
`AIAlgo` requires PyTorch to be installed in the environment to load and use custom trained models. If PyTorch is not available, or if a specified model path is invalid or not found, `AIAlgo` will fall back to a simple, untrained placeholder model for basic operation (not recommended for live trading).

To install PyTorch, refer to the official PyTorch website for instructions suitable for your system: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
A general installation command is:
```bash
pip install torch torchvision torchaudio
```

**Configuration:**
`AIAlgo` is configured within the `config/algo.json` file. Add or modify the `AIAlgo` block as shown below:

```json
{
  "GuppyMMA": { "...": "..." },
  "Bollinger": { "...": "..." },
  "MovingAverageCrossover": { "...": "..." },
  "AIAlgo": {
    "enabled": true,
    "model_path": "models/your_custom_model.pth"
  },
  "maxLost": { "...": "..." },
  "takeProfit": { "...": "..." }
}
```
- `enabled`: Set to `true` to activate `AIAlgo`, `false` to disable it.
- `model_path`: Specifies the path to your trained PyTorch model file. This should be a file containing the model's state dictionary or the entire model saved via `torch.save()`.

**Using a Custom Model:**
You can train your own neural network using PyTorch and integrate it with `AIAlgo`. The model must be compatible with the feature engineering performed by `AIAlgo`.

-   **Input Features**: `AIAlgo` prepares a 1D tensor of floating-point numbers as input for the model. This tensor is a combination of:
    1.  **Price Features**: A sequence of the most recent price points. By default, `AIAlgo` uses 5 price features (4 historical values fetched based on `max_frequencies` and the latest `current_value`).
    2.  **Indicator Signals**: Signals collected from other enabled algorithms. Currently, `AIAlgo` is configured to use signals from:
        *   `GuppyMMA`
        *   `Bollinger`
        *   `MovingAverageCrossover`
        If a signal from one of these indicators is not available (e.g., the algorithm is disabled or encounters an error), `AIAlgo` will use a default value of `0` for that feature.

    The features are combined in the following order: `[price_1, price_2, price_3, price_4, price_5, guppy_signal, bollinger_signal, ma_crossover_signal]`.
    Therefore, a custom model, by default, must expect an input tensor with `5 (prices) + 3 (indicators) = 8` features. If you modify `AIAlgo`'s feature engineering (e.g., `NUM_PRICE_FEATURES` or `TARGET_INDICATOR_KEYS`), your custom model's input layer must match the new feature count and structure.

-   **Output**: The model must output a 1D tensor of 3 values. These values are interpreted as scores/logits for "sell", "hold", and "buy" actions, respectively. `AIAlgo` applies an `argmax` function to this output to determine the final trading signal (-1 for sell, 0 for hold, 1 for buy).

**Model Compatibility is Crucial**: Any custom model loaded via `model_path` **must** be trained using the exact same feature engineering logic (number of features, type of features, and order of features) that `AIAlgo` implements. Mismatches will likely lead to errors or poor performance.

**Example `PlaceholderNet` (used as fallback):**
The internal placeholder model, which reflects the default feature expectations (8 input features), has a structure similar to this:
```python
import torch.nn as nn

class PlaceholderNet(nn.Module):
    def __init__(self, input_size=8, output_size=3): # Reflects 5 price + 3 indicator features
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)
```

## Running the Bot
Execute the main script to start the trading bot:
```bash
python main.py
```
(Assuming `main.py` is the entry point).

## Disclaimer
Trading cryptocurrencies involves significant risk. This software is provided "as is", without warranty of any kind. Use at your own risk.
The developers are not responsible for any financial losses incurred.
