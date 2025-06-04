# pythonTrading
Automated cryptocurrency trading bot.

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
