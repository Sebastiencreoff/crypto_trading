# Python Trading Service

This project is a FastAPI-based **trading-service** designed to manage and execute automated cryptocurrency trading tasks. These tasks, which run individual trading algorithms, are orchestrated as Kubernetes Jobs. The service includes integrated Slack notifications for monitoring and basic interaction.

## Features
- Multiple algorithm support (GuppyMMA, Bollinger Bands, Moving Average Crossover, AI-based).
- Configuration via JSON files.
- Extensible algorithm interface (`AlgoIf`).

## Setup

1.  **Clone the repository.**
2.  **Install dependencies:**
    ```bash
    pip install -e .[test]
    ```
    This command uses `config/pyproject.toml` to build and install the package in editable mode with test dependencies. Run from the project root.
3.  **Database Setup (PostgreSQL):**
    *   The application requires a PostgreSQL database.
    *   For local development with Kubernetes (recommended), refer to the `docs/k3d_guide.md` for detailed instructions on setting up PostgreSQL within a k3d cluster.
    *   General database configuration (connection details) is managed via `config/central_config.json` and can be overridden by environment variables in Kubernetes deployments (see `docs/k3d_guide.md`).
4.  **Configuration:**
    *   Trading parameters, API keys (if not using simulation), and Slack settings are primarily configured in `config/central_config.json` (or its equivalent in a Kubernetes ConfigMap).
    *   The `SLACK_BOT_TOKEN` is expected as an environment variable for the `trading-service`.
    *   Binance API keys are also typically managed via environment variables or Kubernetes secrets if live trading is intended.

## Deployment (Local Development with k3d)

The application is designed to be deployed on Kubernetes. For local development and testing, a lightweight k3d (Kubernetes in Docker) environment is recommended.

**Comprehensive setup instructions are available in: [`docs/k3d_guide.md`](docs/k3d_guide.md)**

This guide covers:
*   Setting up a k3d cluster.
*   Building and loading Docker images (`trading-service`, `trading-task`).
*   Configuring Kubernetes secrets (PostgreSQL, Binance API, Slack).
*   Setting up the ConfigMap for `central_config.json`.
*   Deploying PostgreSQL.
*   Deploying the Trading Service.
*   Setting up RBAC for task management.
*   Verifying the deployment and accessing the service.

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

## Running the Service

The application is run as the `trading-service` deployed on Kubernetes. It is not typically run as a standalone command-line bot anymore.

Please refer to the **[`docs/k3d_guide.md`](docs/k3d_guide.md)** for detailed instructions on deploying and running the service in a local Kubernetes environment. The guide includes steps to build Docker images, set up Kubernetes resources, and interact with the deployed service.

## Slack Integration

The `trading-service` includes integrated Slack notification capabilities. It can send updates about trading activities and can be interacted with via basic Slack commands.

### Setup

1.  **Create a Slack App and Bot User:**
    *   Go to [api.slack.com/apps](https://api.slack.com/apps) and create a new app.
    *   Add a "Bot User" to your app.
    *   Under "OAuth & Permissions", find the "Bot User OAuth Token" (starts with `xoxb-`).
    *   Grant necessary OAuth scopes. For sending messages, `chat:write` is essential. If you plan to extend functionality to read messages or use RTM, add scopes like `app_mentions:read`, `channels:history`, `groups:history`, etc.

2.  **Set Environment Variable:**
    *   The `trading-service` expects the Slack Bot Token to be provided via the `SLACK_BOT_TOKEN` environment variable.
    *   When deploying to Kubernetes, this is typically done by creating a Secret and mounting it as an environment variable in the `trading-service` deployment. Refer to `docs/k3d_guide.md` for an example.

3.  **Configure Default Channel:**
    *   The default Slack channel ID for notifications is configured in `config/central_config.json` (or its ConfigMap equivalent) under the `slack.default_channel_id` key.
    *   To get a channel ID, right-click the channel name in Slack, select "Copy Link". The ID is the last part of the URL (e.g., `C1234567890`).

4.  **Invite Bot to Channel:**
    *   Invite your bot user to the default channel in your Slack workspace.

### Functionality

*   The `trading-service` will send notifications for significant events (e.g., task start/stop, errors, trades) to the configured default Slack channel.
*   The previous Slack command interface (e.g., `status`, `graph` commands via RTM client) that was part of `run_slack_handler.py` is currently **not** active in the `trading-service` by default. The primary interaction model is now through the `trading-service` API. The `send_slack_notification` utility is used internally by the service.
*   A debug API endpoint `/debug/notify` (if enabled in `trading_service/main.py`) can be used to test sending messages via Slack.

### Troubleshooting

*   Check `trading-service` logs for any Slack-related error messages (e.g., "invalid token", "channel not found").
*   Ensure `SLACK_BOT_TOKEN` environment variable is correctly set for the `trading-service` pods.
*   Verify the `default_channel_id` in the configuration is correct and the bot is a member of that channel.
*   Confirm the bot has the necessary OAuth scopes.

## Database Migrations (Alembic)

Alembic is used to manage database schema migrations.
- Migration scripts: `code/alembic/`
- Configuration: `config/alembic.ini` (reads DB connection from `config/central_config.json` or environment variables).

**Running Alembic Commands:**
Ensure `config/central_config.json` (or equivalent environment variables for Kubernetes) points to your database.
From the project root:
```bash
# Example: Create a new migration
alembic -c config/alembic.ini revision -m "your_migration_message"

# Example: Apply all pending migrations
alembic -c config/alembic.ini upgrade head
```
The `config/alembic.ini` is configured to find migration scripts and application models. For Kubernetes deployments, database migrations should typically be run as an Init Container or a separate Job before the main application starts.

## Disclaimer
Trading cryptocurrencies involves significant risk. This software is provided "as is", without warranty of any kind. Use at your own risk.
The developers are not responsible for any financial losses incurred.
