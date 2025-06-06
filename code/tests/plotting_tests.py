import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import datetime

# Assuming crypto_trading is in PYTHONPATH
from crypto_trading.plotting import generate_portfolio_graph, generate_pnl_per_trade_graph

class TestGeneratePortfolioGraph(unittest.TestCase):

    def setUp(self):
        self.valid_data_points = [
            (datetime.datetime(2023, 1, 1, 10, 0, 0), 1000.0),
            (datetime.datetime(2023, 1, 1, 11, 0, 0), 1020.0),
            (datetime.datetime(2023, 1, 1, 12, 0, 0), 1010.0)
        ]
        self.output_path = "test_graph.png"

    def tearDown(self):
        # Clean up the test graph file if it was created by a test that doesn't mock savefig
        # However, most tests will mock plt.savefig, so this might not be strictly necessary.
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

    @patch('crypto_trading.plotting.plt')
    def test_successful_graph_generation(self, mock_plt):
        # Mock the figure and axes objects
        mock_ax = mock_plt.subplots.return_value[1] # subplots() returns (fig, ax)

        result = generate_portfolio_graph(self.valid_data_points, self.output_path)

        mock_plt.subplots.assert_called_once_with(figsize=(10, 6))
        mock_ax.plot.assert_called_once()
        # Could add more specific assertions about the data passed to ax.plot if needed

        mock_ax.set_title.assert_called_once_with('Portfolio Value Over Time')
        mock_ax.set_xlabel.assert_called_once_with('Time')
        mock_ax.set_ylabel.assert_called_once_with('Portfolio Value')
        mock_ax.grid.assert_called_once_with(True)
        mock_plt.xticks.assert_called_once() # Check rotation and ha if needed
        mock_plt.tight_layout.assert_called_once()
        mock_plt.savefig.assert_called_once_with(self.output_path)
        mock_plt.close.assert_called_once_with(mock_plt.subplots.return_value[0]) # Close the fig object

        self.assertEqual(result, self.output_path)

    @patch('crypto_trading.plotting.plt')
    def test_insufficient_data_points_one_point(self, mock_plt):
        data_points = [(datetime.datetime(2023, 1, 1, 10, 0, 0), 1000.0)]
        result = generate_portfolio_graph(data_points, self.output_path)

        self.assertIsNone(result)
        mock_plt.subplots.assert_not_called() # Should not attempt to plot

    @patch('crypto_trading.plotting.plt')
    def test_insufficient_data_points_no_points(self, mock_plt):
        data_points = []
        result = generate_portfolio_graph(data_points, self.output_path)

        self.assertIsNone(result)
        mock_plt.subplots.assert_not_called()

    @patch('crypto_trading.plotting.plt')
    def test_plotting_savefig_exception(self, mock_plt):
        # Mock the figure and axes objects
        mock_fig = mock_plt.subplots.return_value[0]
        mock_ax = mock_plt.subplots.return_value[1]

        mock_plt.savefig.side_effect = Exception("Failed to save file")

        result = generate_portfolio_graph(self.valid_data_points, self.output_path)

        self.assertIsNone(result)
        mock_plt.subplots.assert_called_once() # Plotting setup happens
        mock_plt.savefig.assert_called_once_with(self.output_path) # Attempt to save
        mock_plt.close.assert_called_once_with(mock_fig) # Ensure figure is closed even on error

    @patch('crypto_trading.plotting.plt')
    def test_plotting_general_exception(self, mock_plt):
        # Mock the figure and axes objects
        mock_fig = mock_plt.subplots.return_value[0]
        mock_ax = mock_plt.subplots.return_value[1]

        # Simulate an error during plot formatting, for example
        mock_ax.set_title.side_effect = Exception("Some plotting error")

        result = generate_portfolio_graph(self.valid_data_points, self.output_path)

        self.assertIsNone(result)
        mock_plt.subplots.assert_called_once()
        mock_ax.set_title.assert_called_once() # This call triggers the error
        mock_plt.savefig.assert_not_called() # Save should not be attempted
        mock_plt.close.assert_called_once_with(mock_fig) # Ensure figure is closed


if __name__ == '__main__':
    unittest.main()


class TestGeneratePnlPerTradeGraph(unittest.TestCase):
    def setUp(self):
        self.trades_data_valid = [
            {'label': '2023-01-01 10:00', 'profit': 100.50},
            {'label': '2023-01-02 15:30', 'profit': -50.20},
            {'label': '2023-01-03 12:00', 'profit': 75.00}
        ]
        self.output_path = "test_pnl_graph.png"

    def tearDown(self):
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

    @patch('crypto_trading.plotting.plt')
    def test_successful_pnl_per_trade_graph(self, mock_plt):
        mock_ax = mock_plt.subplots.return_value[1] # ax object

        result = generate_pnl_per_trade_graph(self.trades_data_valid, self.output_path)

        # Check figsize based on number of trades
        num_trades = len(self.trades_data_valid)
        expected_fig_width = max(10, min(25, num_trades * 0.6))
        mock_plt.subplots.assert_called_once_with(figsize=(expected_fig_width, 7))

        mock_ax.bar.assert_called_once()
        # Check colors passed to bar chart
        args, kwargs = mock_ax.bar.call_args
        expected_colors = ['green', 'red', 'green']
        self.assertEqual(kwargs.get('color'), expected_colors)

        mock_ax.set_title.assert_called_once_with('Profit/Loss per Trade')
        mock_ax.set_xlabel.assert_called_once_with('Trade (by Sell Time/ID)')
        mock_ax.set_ylabel.assert_called_once_with('Profit/Loss')
        mock_plt.xticks.assert_called_once_with(rotation=60, ha='right', fontsize=8)
        mock_ax.axhline.assert_called_once_with(0, color='grey', lw=1)
        mock_ax.grid.assert_called_once_with(True, axis='y', linestyle='--', alpha=0.7)
        mock_plt.tight_layout.assert_called_once()
        mock_plt.savefig.assert_called_once_with(self.output_path)
        mock_plt.close.assert_called_once_with(mock_plt.subplots.return_value[0]) # fig object

        self.assertEqual(result, self.output_path)

    @patch('crypto_trading.plotting.plt')
    def test_pnl_per_trade_graph_no_data(self, mock_plt):
        result = generate_pnl_per_trade_graph([], self.output_path)
        self.assertIsNone(result)
        mock_plt.subplots.assert_not_called()

    @patch('crypto_trading.plotting.plt')
    def test_pnl_per_trade_graph_savefig_exception(self, mock_plt):
        mock_fig = mock_plt.subplots.return_value[0]
        mock_plt.savefig.side_effect = Exception("Failed to save PNL file")

        result = generate_pnl_per_trade_graph(self.trades_data_valid, self.output_path)
        self.assertIsNone(result)
        mock_plt.savefig.assert_called_once_with(self.output_path)
        mock_plt.close.assert_called_once_with(mock_fig)

    @patch('crypto_trading.plotting.plt')
    def test_pnl_per_trade_graph_general_exception(self, mock_plt):
        mock_fig = mock_plt.subplots.return_value[0]
        mock_ax = mock_plt.subplots.return_value[1]
        mock_ax.bar.side_effect = Exception("Failed to plot bars")

        result = generate_pnl_per_trade_graph(self.trades_data_valid, self.output_path)
        self.assertIsNone(result)
        mock_ax.bar.assert_called_once() # Error occurs during/after this
        mock_plt.savefig.assert_not_called()
        mock_plt.close.assert_called_once_with(mock_fig)
