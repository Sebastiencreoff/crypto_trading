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

## Slack Integration

The trading bot can be monitored and controlled via Slack messages.

### Setup

To enable Slack integration, you need to configure your Slack Bot User OAuth Token and the target Slack Channel ID in your JSON configuration file (e.g., `config/trading_SIMU.json` or `config/trading_COINBASE.json`).

1.  **Create a Slack App and Bot User:**
    *   Go to [api.slack.com/apps](https://api.slack.com/apps) and create a new app.
    *   Add a "Bot User" to your app.
    *   Under "OAuth & Permissions", find the "Bot User OAuth Token". It usually starts with `xoxb-`. This is your `slack_token`.
    *   Ensure your bot token has the necessary permissions (scopes). Required scopes typically include:
        *   `chat:write`: To send messages.
        *   `app_mentions:read`: To read messages that @mention your bot (if you want to command it outside the designated channel).
        *   `channels:history`, `groups:history`, `im:history`, `mpim:history`: For the RTM client to receive messages in channels and DMs. Generally, scopes like `channels:read` and `groups:read` might also be needed depending on your setup. The RTM client needs to "see" messages.

2.  **Get Channel ID:**
    *   The `slack_channel_id` is the ID of the public channel where the bot will primarily listen and post. You can find this by right-clicking the channel name in Slack and selecting "Copy Link". The ID is the last part of the URL (e.g., `C1234567890`).

3.  **Update Configuration File:**
    Add the following keys to your main JSON configuration file. The `initial_capital` is used as the baseline for the portfolio value graph. If not set, it defaults to `0.0`.
    ```json
    {
      // ... other configurations ...
      "initial_capital": 1000.0, // Your starting capital in the base currency
      "slack_token": "xoxb-your-bot-user-oauth-token-here",
      "slack_channel_id": "C1234567890"
      // ... other configurations ...
    }
    ```
    Replace the placeholder values with your actual token, channel ID, and desired initial capital. If `slack_token` or `slack_channel_id` are missing or `null`, Slack integration will be disabled.

4.  **Invite Bot to Channel:**
    *   Don't forget to invite your bot user to the channel specified by `slack_channel_id` in Slack.

### Usage

Once configured and the bot is running, it will listen for commands in two ways:
1.  Messages posted in the channel specified by `slack_channel_id`.
2.  Direct mentions of the bot (e.g., `@YourBotName command`) in any channel the bot is a member of.

**Available Commands:**

*   `start`: (Currently informational) Checks if the trading bot is running. If the bot was fully stopped, this command cannot restart it from Slack due to the current application design; a manual restart of the application would be needed. It will confirm if the bot is already operational.
*   `stop`: Stops the trading bot. The bot will finish any current processing cycle and then cease further trading activity.
*   `status`: Reports whether the bot is currently running and displays the latest profit/loss figures.
*   `graph`: Generates and uploads a line graph showing the portfolio value over time. The portfolio value is calculated as `Initial Capital + Cumulative Profits` from closed trades.
*   `pnl_chart`: Generates and uploads a bar chart showing the profit or loss for each completed trade, ordered chronologically by sell time.

### Troubleshooting

*   **Bot Not Responding:**
    *   Check the application logs for any error messages related to Slack initialization (e.g., "invalid token", "channel not found") or API communication issues.
    *   Ensure the `slack_token` and `slack_channel_id` in your configuration file are correct.
    *   Verify that the bot user has been invited to the channel specified by `slack_channel_id`.
    *   Confirm the bot has the correct OAuth scopes/permissions in the Slack App settings.

## Disclaimer
Trading cryptocurrencies involves significant risk. This software is provided "as is", without warranty of any kind. Use at your own risk.
The developers are not responsible for any financial losses incurred.
