#!/usr/bin/env python
# main_pipeline.py - Main entry point for the AE-LSTM salinity prediction pipeline

import os
import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

# Import project modules
from src.utils.setup import (
    setup_experiment, 
    setup_logging, 
    setup_seed,
)
from src.data.enhanced_data_loader import EnhancedDataLoader
from src.data.data_processor import DataProcessor
from src.data.sequence_dataset import SequenceDataset
from src.model.autoencoder import AutoencoderModel
from src.model.lstm import LSTMModel
from src.model.ae_lstm import AELSTMModel
from src.train.train_autoencoder import train_autoencoder
from src.train.train_lstm import train_lstm
from src.train.train_ae_lstm import train_ae_lstm
from src.utils.evaluation import evaluate_model
from src.visualization.visualize import (
    plot_reconstruction_samples,
    plot_prediction_samples,
    plot_loss_curves,
    plot_metrics
)

log = logging.getLogger(__name__)

# Add version_base=None
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> float: # Modified return type hint
    """
    Main pipeline for training and evaluating an AE-LSTM model.
    
    The pipeline consists of:
    1. Experiment setup
    2. Data loading and preprocessing
    3. Autoencoder training (or loading pre-trained)
    4. LSTM training on latent codes (or loading pre-trained)
    5. Combined AE-LSTM training/fine-tuning (optional)
    6. Evaluation and visualization
    
    Args:
        cfg: Hydra configuration object
    """
    # 1. Setup experiment
    setup_experiment(cfg)
    setup_seed(cfg.experiment.seed)
    logger = setup_logging(cfg)
    
    # Set device
    device = torch.device(cfg.experiment.device if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    # Initialize TensorBoard if enabled
    if cfg.logging.use_tensorboard:
        tb_dir = os.path.join(cfg.paths.tensorboard_dir, cfg.experiment.name)
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
    else:
        writer = None
    
    # 2. Data loading and preprocessing
    # Use EnhancedDataLoader
    data_loader = EnhancedDataLoader(cfg) 
    data = data_loader.load_data()
    
    data_processor = DataProcessor(cfg)
    processed_data = data_processor.process(data)
    
    # Create datasets for AE
    train_dataset_ae, val_dataset_ae, test_dataset_ae = data_processor.create_ae_datasets(
        processed_data
    )
    
    # 3. Train or load Autoencoder
    autoencoder = AutoencoderModel(cfg).to(device)
    
    if cfg.model.ae_lstm.use_pretrained_ae:
        # Load pretrained autoencoder
        ae_path = os.path.join(cfg.paths.ae_model_dir, "best_model.pt")
        if os.path.exists(ae_path):
            autoencoder.load_state_dict(torch.load(ae_path, map_location=device))
            log.info(f"Loaded pretrained autoencoder from {ae_path}")
        else:
            log.warning(f"No pretrained autoencoder found at {ae_path}. Training new model.")
            train_autoencoder(autoencoder, train_dataset_ae, val_dataset_ae, cfg, device, writer)
    else:
        # Train autoencoder from scratch
        train_autoencoder(autoencoder, train_dataset_ae, val_dataset_ae, cfg, device, writer)
    
    # Generate latent representations for LSTM training
    log.info("Generating latent representations for LSTM training")
    train_latent, val_latent, test_latent = data_processor.generate_latent_representations(
        autoencoder, train_dataset_ae, val_dataset_ae, test_dataset_ae, device
    )
    
    # Create sequence datasets for LSTM
    train_dataset_lstm, val_dataset_lstm, test_dataset_lstm = data_processor.create_sequence_datasets(
        train_latent, val_latent, test_latent, processed_data
    )
    
    # 4. Train or load LSTM
    lstm_model = LSTMModel(cfg).to(device)
    
    if cfg.model.ae_lstm.use_pretrained_lstm:
        # Load pretrained LSTM
        lstm_path = os.path.join(cfg.paths.lstm_model_dir, "best_model.pt")
        if os.path.exists(lstm_path):
            lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
            log.info(f"Loaded pretrained LSTM from {lstm_path}")
        else:
            log.warning(f"No pretrained LSTM found at {lstm_path}. Training new model.")
            train_lstm(lstm_model, train_dataset_lstm, val_dataset_lstm, cfg, device, writer)
    else:
        # Train LSTM from scratch
        train_lstm(lstm_model, train_dataset_lstm, val_dataset_lstm, cfg, device, writer)
    
    # 5. Combined AE-LSTM model (optional fine-tuning)
    ae_lstm_model = AELSTMModel(autoencoder, lstm_model, cfg).to(device)
    
    if cfg.model.ae_lstm.train_ae_end_to_end:
        # Fine-tune the combined model
        log.info("Fine-tuning the combined AE-LSTM model")
        train_ae_lstm(ae_lstm_model, train_dataset_ae, val_dataset_ae, cfg, device, writer)
    
    # 6. Evaluation
    log.info("Evaluating models")
    
    # Evaluate autoencoder reconstruction
    ae_metrics = evaluate_model(
        autoencoder, 
        test_dataset_ae, 
        cfg, 
        device,
        model_type="autoencoder"
    )
    
    # Evaluate LSTM prediction
    lstm_metrics = evaluate_model(
        ae_lstm_model,
        test_dataset_lstm,
        cfg,
        device,
        model_type="ae_lstm"
    )
    
    # 7. Visualization
    log.info("Generating visualizations")
    
    # Plot reconstruction samples
    plot_reconstruction_samples(
        autoencoder,
        test_dataset_ae,
        cfg,
        device,
        n_samples=5
    )
    
    # Plot prediction samples
    plot_prediction_samples(
        ae_lstm_model,
        test_dataset_lstm,
        cfg,
        device,
        n_samples=5
    )
    
    # Plot metrics
    plot_metrics(ae_metrics, lstm_metrics, cfg)
    
    # Close TensorBoard writer
    if writer:
        writer.close()
    
    log.info("Pipeline completed successfully!")
    
    # Return the primary metric for Optuna optimization
    # Assuming we want to minimize the RMSE of the final AE-LSTM prediction
    final_metric = lstm_metrics['metrics'].get('rmse', float('inf')) # Default to infinity if RMSE not found
    log.info(f"Returning metric for Optuna: {final_metric}")
    return final_metric


if __name__ == "__main__":
    main()
