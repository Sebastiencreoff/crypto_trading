import unittest
import logging
import torch
from unittest.mock import patch, MagicMock
from crypto_trading.algo.ai_algo import AIAlgo, PlaceholderNet

class TestAIAlgo(unittest.TestCase):

    def setUp(self):
        self.base_config_dict = {
            "AIAlgo": {
                "enabled": True,
                "model_path": "models/non_existent_model.pth" # Force PlaceholderNet
            },
            # Other algo configs that might be in a real main config,
            # AIAlgo's __init__ takes the whole dict.
            "GuppyMMA": {}, "Bollinger": {}, "MovingAverageCrossover": {}
        }
        # AIAlgo's PlaceholderNet will be initialized with output_size=14
        self.ai_algo = AIAlgo(self.base_config_dict)

        self.current_value = 100.0
        self.sufficient_values = [96.0, 97.0, 98.0, 99.0] # 4 historical
        self.currency = "BTC-USD"
        self.full_indicator_signals = {
            'GuppyMMA': 1,
            'Bollinger': -1,
            'MovingAverageCrossover': 0
        }

    def test_initialization_placeholder_model(self):
        """Test that AIAlgo initializes with PlaceholderNet when model_path is invalid."""
        self.assertIsInstance(self.ai_algo.model, PlaceholderNet)
        # PlaceholderNet is hardcoded to input_size=8, output_size=14 in AIAlgo if no model loaded
        # Expected input size for AIAlgo is 5 price features + 3 indicator features = 8
        self.assertEqual(self.ai_algo.model.linear.in_features, self.ai_algo.expected_input_size)
        self.assertEqual(self.ai_algo.model.linear.out_features, AIAlgo.MODEL_OUTPUT_SIZE) # Should be 14

    @patch.object(AIAlgo, 'process') # Mock process to simplify testing __init__ behavior with loaded model
    def test_initialization_with_mocked_loadable_model(self, mock_process):
        """Test AIAlgo initialization if a model were successfully loaded."""

        # Create a mock model that mimics a loaded PyTorch model
        mock_model_instance = MagicMock(spec=torch.nn.Module)
        # Simulate the model having a linear layer to check input/output features
        mock_linear_layer = MagicMock(spec=torch.nn.Linear)
        mock_linear_layer.in_features = self.ai_algo.expected_input_size
        mock_linear_layer.out_features = AIAlgo.MODEL_OUTPUT_SIZE

        # Make the mock model's modules() return an iterable containing the mock_linear_layer
        def modules_side_effect():
            yield mock_linear_layer

        mock_model_instance.modules = MagicMock(side_effect=modules_side_effect)

        config_with_valid_path = {
            "AIAlgo": {"model_path": "dummy_valid_path.pth"},
            "GuppyMMA": {}, "Bollinger": {}, "MovingAverageCrossover": {}
        }

        with patch('os.path.exists') as mock_exists, \
             patch('torch.load') as mock_torch_load:

            mock_exists.return_value = True # Simulate model file exists
            mock_torch_load.return_value = mock_model_instance # Return our mock model

            ai_algo_loaded = AIAlgo(config_with_valid_path)

            mock_torch_load.assert_called_with("dummy_valid_path.pth")
            self.assertEqual(ai_algo_loaded.model, mock_model_instance)
            # Check if eval was called
            mock_model_instance.eval.assert_called_once()


    def test_process_output_structure_and_scaling(self):
        """
        Test AIAlgo.process for correct output structure, AI signal,
        and parameter scaling/transformation for other algos.
        """
        # Mock the model's output tensor (14 values)
        # Values chosen to test various scaling/clamping aspects
        mock_output_tensor = torch.tensor([
            0.1, 0.8, 0.3,  # AI Signal (Buy: index 1)
            4.5,            # Guppy: num_short (raw) -> 4
            2.0,            # Guppy: start_short (raw) -> 3 (clamped by min)
            1.2,            # Guppy: step_short (raw) -> 1
            7.0,            # Guppy: num_long (raw) -> 6 (clamped by max)
            25.0,           # Guppy: start_long (raw) -> 30 (clamped by min)
            12.0,           # Guppy: step_long (raw) -> 10 (clamped by max)
            0.7,            # Guppy: buy_factor (logit) -> sigmoid -> ~0.66, threshold = int(0.66*4)=2
            -0.4,           # Guppy: sell_factor (logit) -> sigmoid -> ~0.40, threshold = int(0.40*4)=1
            15.0,           # Bollinger: frequency (raw) -> 15
            8.0,            # MAC: short_window (raw) -> 8
            3.0             # MAC: long_offset (raw) -> 5 (clamped by min for offset)
        ], dtype=torch.float32)

        # Patch the model instance within ai_algo
        self.ai_algo.model = MagicMock(return_value=mock_output_tensor)

        ai_signal, algo_configs = self.ai_algo.process(
            self.current_value, self.sufficient_values, self.currency, self.full_indicator_signals
        )

        # 1. Test AI own signal
        self.assertEqual(ai_signal, 1, "AI own signal should be Buy (1)")

        # 2. Test algo_configs_map structure
        self.assertIsInstance(algo_configs, dict)
        self.assertIn('GuppyMMA', algo_configs)
        self.assertIn('Bollinger', algo_configs)
        self.assertIn('MovingAverageCrossover', algo_configs)

        # 3. Test GuppyMMA config
        guppy_cfg = algo_configs['GuppyMMA']
        self.assertEqual(guppy_cfg['short_term'], [3, 4, 5, 6]) # start=3, step=1, num=4
        self.assertEqual(guppy_cfg['long_term'], [30, 40, 50, 50, 50, 50]) # start=30, step=10, num=6 (last elements will be clamped by max long_term_step effect)
                                                                        # Corrected: list generation is start + i * step.
                                                                        # Expected: [30, 30+10, 30+2*10, 30+3*10, 30+4*10, 30+5*10] -> [30,40,50,60,70,80] - this was previous understanding
                                                                        # The scaling is applied to params *before* list generation
                                                                        # num_long_terms = 6, long_term_start = 30, long_term_step = 10
                                                                        # So, [30, 40, 50, 60, 70, 80]
        # Re-evaluating Guppy long_terms based on code:
        # raw_num_long = 7.0 -> num_long_terms = 6 (clamped)
        # raw_start_long = 25.0 -> long_term_start = 30 (clamped)
        # raw_step_long = 12.0 -> long_term_step = 10 (clamped)
        # Expected: [30, 30+10, 30+2*10, 30+3*10, 30+4*10, 30+5*10] = [30, 40, 50, 60, 70, 80]
        self.assertEqual(guppy_cfg['long_term'], [30, 40, 50, 60, 70, 80])


        # buy_factor = sigmoid(0.7) approx 0.668. num_short_terms = 4. buy_threshold = int(0.668 * 4) = int(2.672) = 2. (max(1,2)=2)
        self.assertEqual(guppy_cfg['buy'], 2)
        # sell_factor = sigmoid(-0.4) approx 0.401. num_short_terms = 4. sell_threshold = int(0.401*4) = int(1.604) = 1. (max(1,1)=1)
        self.assertEqual(guppy_cfg['sell'], 1)


        # 4. Test Bollinger config
        bollinger_cfg = algo_configs['Bollinger']
        self.assertEqual(bollinger_cfg['frequency'], 15) # Clamped: max(5, min(100, int(15.0)))

        # 5. Test MovingAverageCrossover config
        mac_cfg = algo_configs['MovingAverageCrossover']
        self.assertEqual(mac_cfg['short_window'], 8)  # Clamped: max(5, min(100, int(8.0)))
        # long_offset = max(5, min(100, int(3.0))) = 5
        # long_window = short_window (8) + long_offset (5) = 13
        self.assertEqual(mac_cfg['long_window'], 13)
        self.assertTrue(mac_cfg['short_window'] < mac_cfg['long_window'])

    def test_process_error_in_model(self):
        """Test AIAlgo.process when the model call raises an exception."""
        self.ai_algo.model = MagicMock(side_effect=Exception("Model inference error"))

        ai_signal, algo_configs = self.ai_algo.process(
            self.current_value, self.sufficient_values, self.currency, self.full_indicator_signals
        )
        self.assertEqual(ai_signal, 0, "AI signal should be neutral (0) on error")
        self.assertEqual(algo_configs, {}, "Algo configs should be empty on error")

    def test_process_feature_mismatch(self):
        """Test AIAlgo.process when feature count does not match model expectation."""
        # Temporarily change expected_input_size to force a mismatch
        original_expected_size = self.ai_algo.expected_input_size
        self.ai_algo.expected_input_size = 99

        with patch.object(logging, 'error') as mock_log_error:
            ai_signal, algo_configs = self.ai_algo.process(
                self.current_value, self.sufficient_values, self.currency, self.full_indicator_signals
            )

        self.assertEqual(ai_signal, 0)
        self.assertEqual(algo_configs, {})
        mock_log_error.assert_called_once()
        self.assertTrue("Feature mismatch" in mock_log_error.call_args[0][0])

        self.ai_algo.expected_input_size = original_expected_size # Restore

    def test_get_target_algo_configs(self):
        """Tests the get_target_algo_configs method."""
        configs = self.ai_algo.get_target_algo_configs()
        self.assertIsInstance(configs, dict)
        expected_keys = ['GuppyMMA', 'Bollinger', 'MovingAverageCrossover']
        for key in expected_keys:
            self.assertIn(key, configs)
            self.assertIsInstance(configs[key], dict)
        # Check one specific default to ensure it's loading the hardcoded defaults
        self.assertEqual(configs['Bollinger']['frequency'], 20) # Default in AIAlgo's __init__ for its own reference

if __name__ == '__main__':
    unittest.main()
