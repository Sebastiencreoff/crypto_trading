import numpy as np
import logging

def calculate_log_return_volatility(price_series: list[float], window: int = 20) -> float | None:
    """
    Calculates the annualized rolling standard deviation of logarithmic returns.

    Args:
        price_series (list[float]): A list of prices, ordered from oldest to newest.
        window (int): The rolling window period for volatility calculation.

    Returns:
        float | None: The annualized volatility, or None if calculation is not possible
                     (e.g., not enough data).
    """
    if not isinstance(price_series, list) or not all(isinstance(p, (int, float)) for p in price_series):
        logging.error("Volatility: price_series must be a list of numbers.")
        return None

    if len(price_series) < window + 1:
        logging.warning(
            f"Volatility: Not enough data to calculate volatility for window {window}. "
            f"Need {window + 1} prices, got {len(price_series)}."
        )
        return None

    if window <= 0:
        logging.error("Volatility: Window period must be positive.")
        return None

    try:
        prices = np.array(price_series, dtype=float)
        # Calculate logarithmic returns
        # log_returns = np.log(prices[1:] / prices[:-1])
        # A more robust way to calculate log returns to avoid issues if prices are 0 or negative,
        # and to handle cases where prices[t-1] might be zero.
        # We will filter out non-positive prices before taking logs.

        # Ensure prices are positive for log returns
        if np.any(prices <= 0):
            logging.warning("Volatility: Price series contains non-positive values. Cannot calculate log returns accurately.")
            # Attempt to use only the part of the series that is positive if possible
            # This might not be ideal, but better than failing outright if some leading values are bad.
            # However, for simplicity and correctness, we will return None if non-positive prices are found
            # where log returns would be computed.
            # We need at least two prices to calculate one return.
            # Check from prices[-(window+1):] which are relevant for the last volatility calculation.
            relevant_prices_for_log = prices[-(window + 1):]
            if np.any(relevant_prices_for_log <=0):
                logging.error("Volatility: Relevant prices for log return calculation include non-positive values.")
                return None

        log_returns = np.log(prices[1:] / prices[:-1])

        # Calculate rolling standard deviation of log returns
        # We need `window` number of log returns, which means `window+1` prices.
        # The `log_returns` array will have `len(prices)-1` elements.
        # We take the last `window` returns to calculate std dev.
        if len(log_returns) < window:
            logging.warning(
                f"Volatility: Not enough log returns ({len(log_returns)}) "
                f"to calculate volatility for window {window}."
            )
            return None

        relevant_log_returns = log_returns[-window:]
        std_dev_log_returns = np.std(relevant_log_returns)

        # Annualize the volatility (assuming daily data if not specified otherwise, common practice)
        # Standard is 252 trading days in a year.
        # If data is hourly, this would be different (252 * 24), etc.
        # For now, let's assume the caller handles the interpretation of 'period'
        # and we provide a non-annualized standard deviation of log returns for the given window.
        # Or, we can make annualization_factor a parameter.
        # For simplicity, returning the direct standard deviation of the log returns for the window.
        # annualization_factor = 252 # Example for daily data
        # annualized_volatility = std_dev_log_returns * np.sqrt(annualization_factor)

        # Returning the direct standard deviation of the log returns for the window.
        # The interpretation (daily, hourly) and annualization is up to the caller or further refinement.
        volatility = std_dev_log_returns

        logging.debug(f"Calculated volatility: {volatility} for window {window} with {len(price_series)} prices.")
        return volatility

    except Exception as e:
        logging.error(f"Volatility: Error during calculation: {e}")
        return None

def example_usage():
    # Example of how to use it
    logging.basicConfig(level=logging.DEBUG)
    prices1 = [100, 102, 101, 103, 105, 104, 106, 107, 108, 110, 109, 112, 111, 113, 115, 114, 116, 117, 118, 120, 119] # 21 prices
    vol1 = calculate_log_return_volatility(prices1, window=20) # Needs 21 prices for 20 returns
    logging.info(f"Example Volatility 1 (20-day window on 21 prices): {vol1}")

    prices2 = list(np.random.lognormal(mean=np.log(100), sigma=0.01, size=50)) # 50 prices
    vol2 = calculate_log_return_volatility(prices2, window=20)
    logging.info(f"Example Volatility 2 (20-day window on 50 prices): {vol2}")

    prices_short = [100, 101, 102] # 3 prices
    vol_short = calculate_log_return_volatility(prices_short, window=2) # Needs 3 prices for 2 returns
    logging.info(f"Example Volatility Short (2-day window on 3 prices): {vol_short}")

    vol_short_fail = calculate_log_return_volatility(prices_short, window=3) # Not enough
    logging.info(f"Example Volatility Short Fail (3-day window on 3 prices): {vol_short_fail}")

    prices_zero = [100, 0, 102]
    vol_zero = calculate_log_return_volatility(prices_zero, window=2)
    logging.info(f"Example Volatility with zero price: {vol_zero}")

    prices_neg = [100, -10, 102]
    vol_neg = calculate_log_return_volatility(prices_neg, window=2)
    logging.info(f"Example Volatility with negative price: {vol_neg}")


if __name__ == '__main__':
    # This block will run if the script is executed directly
    # Useful for testing the module
    example_usage()
