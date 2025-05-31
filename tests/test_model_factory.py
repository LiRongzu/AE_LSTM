import unittest
from omegaconf import OmegaConf
from src.model.factory import get_model
from src.model.lstm import LSTMModel # Assuming this is the standard one
from src.model.mamba import MambaModel
from src.model.transformer import TransformerModel

class TestModelFactory(unittest.TestCase):
    def setUp(self):
        # Base config structure, specific model configs will be merged/selected by get_model
        # The get_model function will receive a cfg where cfg.model.name determines which section to use.
        # The individual sections (lstm, mamba, transformer) should contain all necessary model-specific args.
        self.cfg = OmegaConf.create({
            'model': {
                # 'name' will be set in each test method
                'autoencoder': {'latent_dim': 32}, # Example, not directly used by get_model but good for context

                'lstm': {
                    # Parameters for LSTMModel
                    'name': 'lstm', # Often good practice to have name within the section too
                    'type': 'standard', # For LSTM variants
                    'input_size': 32,
                    'hidden_size': 64,
                    'num_layers': 1,
                    'output_size': 32,
                    'dropout': 0.1,
                    'sequence_length': 10 # Not directly used by LSTMModel constructor but part of its config block
                },
                'mamba': {
                    # Parameters for MambaModel
                    'name': 'mamba',
                    'input_size': 32, # Will be d_model for Mamba core if no separate projection
                    'output_size': 32,
                    'd_model': 32,    # Mamba's internal dimension
                    'n_layer': 2,     # Number of layers
                    'dt_rank': 'auto',
                    'd_conv': 4,
                    'conv_bias': True,
                    'd_state': 16,
                    'expand': 2,
                    'bias': False,
                    'dropout': 0.1, # Added dropout to match MambaModel's MinimalMambaArgs expectation
                                     # (though it's applied in ResidualBlock, not MambaBlock directly)
                    # Other params like dt_min etc. are part of full config but not all are mandatory for MinimalMambaArgs
                    'dt_min': 0.001,
                    'dt_max': 0.1,
                    'dt_init': "random",
                    'dt_scale': 1.0,
                    'dt_init_floor': 1e-4,
                    'use_fast_path': False,
                    'layer_idx': None,

                },
                'transformer': {
                    # Parameters for TransformerModel
                    'name': 'transformer',
                    'input_size': 32,
                    'output_size': 32,
                    'd_model': 64,
                    'nhead': 4,
                    'num_encoder_layers': 2,
                    'dim_feedforward': 128,
                    'dropout': 0.1,
                    'sequence_length': 10
                }
            }
            # train configurations would also be needed if models load them upon init, but current models don't
        })

    def test_get_lstm_model(self):
        current_cfg = self.cfg.copy()
        current_cfg.model.name = 'lstm'
        # Simulate how input_size/output_size might be set in main_pipeline
        current_cfg.model.lstm.input_size = current_cfg.model.autoencoder.latent_dim
        current_cfg.model.lstm.output_size = current_cfg.model.autoencoder.latent_dim
        model = get_model(current_cfg)
        self.assertIsInstance(model, LSTMModel)

    def test_get_mamba_model(self):
        current_cfg = self.cfg.copy()
        current_cfg.model.name = 'mamba'
        current_cfg.model.mamba.input_size = current_cfg.model.autoencoder.latent_dim
        current_cfg.model.mamba.output_size = current_cfg.model.autoencoder.latent_dim
        # For Mamba, d_model is key. If input_size from AE is to be used as d_model:
        current_cfg.model.mamba.d_model = current_cfg.model.autoencoder.latent_dim
        model = get_model(current_cfg)
        self.assertIsInstance(model, MambaModel)

    def test_get_transformer_model(self):
        current_cfg = self.cfg.copy()
        current_cfg.model.name = 'transformer'
        current_cfg.model.transformer.input_size = current_cfg.model.autoencoder.latent_dim
        current_cfg.model.transformer.output_size = current_cfg.model.autoencoder.latent_dim
        model = get_model(current_cfg)
        self.assertIsInstance(model, TransformerModel)

    def test_invalid_model_name(self):
        current_cfg = self.cfg.copy()
        current_cfg.model.name = 'invalid_model_type'
        with self.assertRaises(ValueError):
            get_model(current_cfg)

if __name__ == '__main__':
    unittest.main()
