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
