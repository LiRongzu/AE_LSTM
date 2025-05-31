import unittest
import torch
from omegaconf import OmegaConf # Ensure OmegaConf is imported
from src.model.mamba import MambaModel # Adjust import if MambaModel is elsewhere

class TestMambaModel(unittest.TestCase):
    def setUp(self):
        # Create a dummy DictConfig for Mamba model
        # Ensure these params match what MambaModel expects from its config
        self.cfg = OmegaConf.create({
            'model': {
                'name': 'mamba', # Added model name for consistency, though MambaModel itself might not use it directly
                'mamba': {
                    'input_size': 32, # This will be d_model for the Mamba core
                    'output_size': 32, # For the final output layer
                    'd_model': 32, # d_model should match input_size for this setup
                    'n_layer': 2,
                    'dt_rank': 'auto',
                    'dt_min': 0.001, # These extra params are in config but not all used by MinimalMambaArgs
                    'dt_max': 0.1,
                    'dt_init': 'random',
                    'dt_scale': 1.0,
                    'dt_init_floor': 1e-4,
                    'd_conv': 4,
                    'conv_bias': True,
                    'd_state': 16,
                    'expand': 2,
                    'use_fast_path': False, # Not used by minimal implementation
                    'layer_idx': None,      # Not used by minimal implementation
                    'dropout': 0.1,         # Used by MinimalMambaArgs via self.args in ResidualBlock, but not directly in MambaBlock
                    'bias': False,          # Used by MinimalMambaArgs for linear layers
                }
            }
        })
        # The MambaModel's internal d_model is taken from cfg.model.mamba.d_model
        # The input to the MambaModel's forward pass should have feature dim == d_model
        self.model = MambaModel(self.cfg)
        self.batch_size = 4
        self.seq_len = 10
        # Input to the model will have feature size d_model
        self.input_feature_size = self.cfg.model.mamba.d_model

    def test_model_initialization(self):
        self.assertIsNotNone(self.model)

    def test_forward_pass(self):
        # dummy_input shape: (batch_size, seq_len, d_model)
        dummy_input = torch.randn(self.batch_size, self.seq_len, self.input_feature_size)
        output = self.model(dummy_input)

        # MambaModel is now designed to output (batch_size, output_size) from the last token
        self.assertEqual(output.shape, (self.batch_size, self.cfg.model.mamba.output_size))

if __name__ == '__main__':
    unittest.main()
