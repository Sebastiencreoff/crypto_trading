import matplotlib
matplotlib.use('Agg') # Use Agg backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import logging # For logging errors

def generate_portfolio_graph(data_points: list, output_path: str) -> str | None:
    """
    Generates a line graph of portfolio value over time and saves it to a file.

    Args:
        data_points: A list of (timestamp, value) tuples.
        output_path: The file path to save the generated PNG image.

    Returns:
        The output_path if the graph was generated and saved successfully, None otherwise.
    """
    if not data_points or len(data_points) < 2:
        logging.warning("Not enough data points to generate a meaningful graph.")
        # Optionally, could create a graph with "No data" text, but returning None is simpler for now.
        return None

    try:
        timestamps = [point[0] for point in data_points]
        values = [point[1] for point in data_points]

        fig, ax = plt.subplots(figsize=(10, 6)) # Adjust figure size as needed

        ax.plot(timestamps, values, marker='o', linestyle='-')

        # Formatting the plot
        ax.set_title('Portfolio Value Over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Portfolio Value') # Assuming currency is handled by context or config

        # Format the x-axis to handle datetime objects nicely
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45, ha='right') # Rotate labels for better readability

        ax.grid(True)
        plt.tight_layout() # Adjust layout to prevent labels from being cut off

        plt.savefig(output_path)
        plt.close(fig) # Close the figure to free up memory

        logging.info(f"Portfolio graph saved to {output_path}")
        return output_path

    except Exception as e:
        logging.error(f"Error generating portfolio graph: {e}", exc_info=True)
        # Ensure the figure is closed if an error occurs after it's created
        if 'fig' in locals() and fig:
            plt.close(fig)
        return None


def generate_pnl_per_trade_graph(trades_data: list, output_path: str) -> str | None:
    """
    Generates a bar chart of Profit/Loss per trade and saves it to a file.

    Args:
        trades_data: A list of dictionaries, where each dictionary has 'label' (str)
                     and 'profit' (float). Example:
                     [{'label': '2023-01-01 10:00', 'profit': 100.50}, ...]
        output_path: The file path to save the generated PNG image.

    Returns:
        The output_path if the graph was generated and saved successfully, None otherwise.
    """
    if not trades_data:
        logging.warning("No trades data to generate PnL per trade graph.")
        return None

    try:
        labels = [item['label'] for item in trades_data]
        profits = [item['profit'] for item in trades_data]

        bar_colors = ['green' if p >= 0 else 'red' for p in profits]

        # Adjust figsize: width might need to be larger for many trades
        # For example, 0.5 inches per trade, minimum 10 inches. Max 20-25 to avoid huge images.
        num_trades = len(labels)
        fig_width = max(10, min(25, num_trades * 0.6))
        fig_height = 7 # Keep height somewhat constant or adjust based on label rotation needs

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        ax.bar(labels, profits, color=bar_colors)

        # Formatting the plot
        ax.set_title('Profit/Loss per Trade')
        ax.set_xlabel('Trade (by Sell Time/ID)')
        ax.set_ylabel('Profit/Loss') # Assuming currency is consistent

        # Rotate X-axis labels for better readability
        plt.xticks(rotation=60, ha='right', fontsize=8) # Increased rotation for potentially long labels

        ax.axhline(0, color='grey', lw=1) # Horizontal line at y=0
        ax.grid(True, axis='y', linestyle='--', alpha=0.7) # Grid for y-axis

        plt.tight_layout() # Adjust layout to prevent labels from being cut off

        plt.savefig(output_path)
        plt.close(fig) # Close the figure to free up memory

        logging.info(f"PnL per trade graph saved to {output_path}")
        return output_path

    except Exception as e:
        logging.error(f"Error generating PnL per trade graph: {e}", exc_info=True)
        if 'fig' in locals() and fig: # Ensure fig is defined before trying to close
            plt.close(fig)
        return None


if __name__ == '__main__':
    # Example usage (for testing the function directly)
    print("Plotting example (requires data and output path):")
    # Create some dummy data for a quick test
    # Note: This example won't run in the agent's environment directly without a place to save.
    # It's here for conceptual testing.
    # dummy_data = [
    #     (datetime.datetime(2023, 1, 1, 10, 0, 0), 1000),
    #     (datetime.datetime(2023, 1, 1, 11, 0, 0), 1020),
    #     (datetime.datetime(2023, 1, 1, 12, 0, 0), 1010),
    #     (datetime.datetime(2023, 1, 1, 13, 0, 0), 1050),
    #     (datetime.datetime(2023, 1, 1, 14, 0, 0), 1080),
    # ]
    # test_output_path = "portfolio_graph_test.png"
    #
    # if len(dummy_data) < 2:
    #     print("Dummy data is insufficient for plotting example.")
    # else:
    #     # Need to ensure matplotlib is installed if running this block locally
    #     try:
    #         import matplotlib
    #         print(f"Matplotlib version: {matplotlib.__version__}")
    #         result_path = generate_portfolio_graph(dummy_data, test_output_path)
    #         if result_path:
    #             print(f"Test graph generated at: {result_path}")
    #         else:
    #             print("Test graph generation failed.")
    #     except ImportError:
    #         print("Matplotlib not installed, skipping plotting example.")
    #     except Exception as e:
    #         print(f"An error occurred during example plotting: {e}")
