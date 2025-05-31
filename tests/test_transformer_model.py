import unittest
import torch
from omegaconf import OmegaConf # Ensure OmegaConf is imported
from src.model.transformer import TransformerModel

class TestTransformerModel(unittest.TestCase):
    def setUp(self):
        self.cfg = OmegaConf.create({
            'model': {
                'name': 'transformer', # Added model name for consistency
                'transformer': {
                    'input_size': 32,    # Feature dimension of input sequence
                    'output_size': 32,   # Desired output dimension
                    'd_model': 64,       # Internal dimension of the transformer (embedding dim)
                    'nhead': 4,          # Number of attention heads
                    'num_encoder_layers': 2,
                    'dim_feedforward': 128,
                    'dropout': 0.1,
                    'sequence_length': 10 # Max sequence length, used by PositionalEncoding
                }
            }
        })
        self.model = TransformerModel(self.cfg)
        self.batch_size = 4
        # sequence_length for input should match what PositionalEncoding is configured for,
        # or at least be <= max_len of PositionalEncoding.
        # The TransformerModel itself can handle variable seq_len up to max_len of PE.
        self.seq_len = self.cfg.model.transformer.sequence_length
        self.input_size = self.cfg.model.transformer.input_size

    def test_model_initialization(self):
        self.assertIsNotNone(self.model)

    def test_forward_pass(self):
        dummy_input = torch.randn(self.batch_size, self.seq_len, self.input_size)
        output = self.model(dummy_input)
        # TransformerModel is designed to output (batch_size, output_size) from the last token
        self.assertEqual(output.shape, (self.batch_size, self.cfg.model.transformer.output_size))

if __name__ == '__main__':
    unittest.main()
