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
You can train your own neural network using PyTorch and integrate it with `AIAlgo`. The model should adhere to the following input/output specifications:

-   **Input**: The model should accept a 1D tensor of floating-point numbers representing a sequence of recent prices. By default, `AIAlgo` prepares the most recent 5 price points (4 historical values + the current value) for the model. If your model requires a different input size, you will need to adjust the data preparation step in `AIAlgo.process()` or ensure your model's `input_size` matches this.
-   **Output**: The model must output a 1D tensor of 3 values. These values are interpreted as scores/logits for "sell", "hold", and "buy" actions, respectively. `AIAlgo` applies an `argmax` function to this output to determine the final trading signal (-1 for sell, 0 for hold, 1 for buy).

**Example `PlaceholderNet` (used as fallback):**
The internal placeholder model has a structure similar to this:
```python
import torch.nn as nn

class PlaceholderNet(nn.Module):
    def __init__(self, input_size=5, output_size=3):
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
=======
## Installation

This project uses Python and dependencies are listed in `setup.py`.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Sebastiencreoff/pythonTrading.git
    cd pythonTrading
    ```

2.  **Install the package and dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
    Then, install the package:
    ```bash
    pip install .
    ```
    This will install all necessary dependencies, including Streamlit for the configuration UI. If you only need to run the Streamlit app and prefer not to install the whole package, ensure Streamlit is installed:
    ```bash
    pip install streamlit
    ```

## Usage

### Running the Trading Bot
The main trading bot can be executed as a console script (details to be added or refer to existing project documentation).

### Configuration Front Page
A Streamlit application is available for easier configuration of the trading bot. This allows you to modify settings for trading modes (Coinbase, Simulation) and algorithm parameters.

**To run the configuration front page:**

1.  Ensure you are in the root directory of the project.
2.  Make sure Streamlit is installed (see Installation section).
3.  Run the following command in your terminal:
    ```bash
    streamlit run app.py
    ```
This will open the configuration interface in your web browser. Changes made in the UI can be saved back to the JSON configuration files in the `config/` directory.

## Contributing
(Information about contributing can be added here)

## License
This project is licensed under the MIT License - see the LICENSE file for details.

