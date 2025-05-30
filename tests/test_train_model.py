import unittest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from omegaconf import OmegaConf
import os
import shutil # For cleaning up directories
import logging

# Configure basic logging for tests if needed, or rely on train_model's logging
logging.basicConfig(level=logging.INFO)

from src.train.train_model import train_model

# Dummy model for testing train_model
class DummyPredictiveModel(nn.Module):
    def __init__(self, input_size, output_size, seq_len_last=True): # seq_len_last indicates if model processes sequence
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.seq_len_last = seq_len_last # If true, model expects (B, S, F) and outputs (B, F) from last step

    def forward(self, x):
        # Simulate taking the last step if input is (batch, seq, feature)
        # and model is designed to process sequence and output based on last step
        if x.ndim == 3 and self.seq_len_last:
            x = x[:, -1, :] # Take last time step
        return self.linear(x)

class TestTrainModel(unittest.TestCase):
    def setUp(self):
        self.model_name = 'dummy_model' # Generic name for the dummy
        self.output_dir = "./test_train_model_output_temp" # Temp dir for test outputs

        # Clean up before test if directory exists
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.cfg = OmegaConf.create({
            'model': {
                'name': self.model_name, # This name will be used to find configs below
                self.model_name: {
                    'input_size': 10, # Corresponds to feature size for the dummy model
                    'output_size': 5,
                    # No other architectural params needed for DummyPredictiveModel
                }
            },
            'train': {
                self.model_name: {
                    'epochs': 1,
                    'batch_size': 2, # This will be used by main_pipeline to create DataLoaders
                    'learning_rate': 0.01,
                    'weight_decay': 1e-5,
                    'optimizer': {'type': 'Adam'},
                    'scheduler': {'factor': 0.1, 'patience': 5, 'min_lr': 1e-6},
                    'early_stopping': {'patience': 3, 'min_delta': 0.001},
                    'checkpoint_frequency': 1, # Ensure checkpointing is tested
                },
                # General train configs (can be overridden by model-specific ones if train_model supports it)
                'checkpoint_frequency': 1, # Example: general checkpoint freq from prompt for train_model
                'log_interval': 1, # Example
            },
            'paths': {
                'output_dir': self.output_dir,
                # train_model will construct model_dir like: cfg.paths.get(f"{model_name}_model_dir", os.path.join(cfg.paths.model_dir, model_name))
                # So, we need model_dir defined.
                'model_dir': os.path.join(self.output_dir, 'models'), # Base model directory
                # Specific directory for this dummy model's artifacts, as constructed by train_model
                f'{self.model_name}_model_dir': os.path.join(self.output_dir, 'models', self.model_name),
                'tensorboard_dir': os.path.join(self.output_dir, 'tensorboard'), # Not used if writer is None
            },
            'experiment': {'device': 'cpu', 'name': 'test_experiment'},
            'logging': {'use_tensorboard': False, 'log_batch_metrics': False}
        })

        # Ensure model save directory exists, as train_model expects it
        os.makedirs(self.cfg.paths[f'{self.model_name}_model_dir'], exist_ok=True)

        self.dummy_model_instance = DummyPredictiveModel(
            self.cfg.model[self.model_name].input_size,
            self.cfg.model[self.model_name].output_size
        ).to(torch.device(self.cfg.experiment.device))

        # Dummy data and loaders
        # Predictive models take (Batch, Seq_len, Features)
        # Targets are (Batch, Features) as models predict last step
        dummy_train_data = torch.randn(4, 3, self.cfg.model[self.model_name].input_size) # 4 samples, seq_len 3
        dummy_train_targets = torch.randn(4, self.cfg.model[self.model_name].output_size)
        dummy_val_data = torch.randn(2, 3, self.cfg.model[self.model_name].input_size) # 2 samples, seq_len 3
        dummy_val_targets = torch.randn(2, self.cfg.model[self.model_name].output_size)

        # Batch size for DataLoader comes from train config, consistent with main_pipeline
        train_batch_size = self.cfg.train[self.model_name].batch_size
        self.train_loader = DataLoader(TensorDataset(dummy_train_data, dummy_train_targets), batch_size=train_batch_size)
        self.val_loader = DataLoader(TensorDataset(dummy_val_data, dummy_val_targets), batch_size=train_batch_size)
        self.device = torch.device(self.cfg.experiment.device)

    def test_train_model_single_epoch(self):
        # Test if train_model runs for one epoch without crashing
        trained_model = train_model(
            self.dummy_model_instance,
            self.train_loader,
            self.val_loader,
            self.cfg,
            self.device,
            writer=None
        )
        self.assertIsNotNone(trained_model)
        # Check if model files were created (best_model.pt, final_model.pt, training_history.pt)
        model_dir = self.cfg.paths[f"{self.model_name}_model_dir"]
        self.assertTrue(os.path.exists(os.path.join(model_dir, "best_model.pt")))
        self.assertTrue(os.path.exists(os.path.join(model_dir, "final_model.pt")))
        self.assertTrue(os.path.exists(os.path.join(model_dir, "training_history.pt")))

    def tearDown(self):
        # Clean up created directories
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

if __name__ == '__main__':
    unittest.main()
