#!/usr/bin/env python
# main_pipeline.py - Main entry point for the AE-LSTM salinity prediction pipeline

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0 = all logs, 1 = filter 
import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import gc  # Add garbage collection
import psutil  # Add process monitoring
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

log = logging.getLogger(__name__)

def log_memory_usage(stage: str):
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()
    log.info(f"[{stage}] Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB ({memory_percent:.1f}%)")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_memory_max = torch.cuda.max_memory_allocated() / 1024 / 1024
        log.info(f"[{stage}] GPU memory: {gpu_memory:.1f} MB (max: {gpu_memory_max:.1f} MB)")

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
    # 1. Setup experiment——————————————————————————————————————————
    setup_experiment(cfg)
    setup_seed(cfg.experiment.seed)
    logger = setup_logging(cfg)
    
    # Set device
    device = torch.device(cfg.experiment.device if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    log_memory_usage("Pipeline start")
    
    # Initialize TensorBoard if enabled
    if cfg.logging.use_tensorboard:
        tb_dir = os.path.join(cfg.paths.tensorboard_dir, cfg.experiment.name)
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
    else:
        writer = None
    
    # 2. Data loading and preprocessing——————————————————————————————————
    # Use EnhancedDataLoader
    data_loader = EnhancedDataLoader(cfg) 
    data = data_loader.load_data()
    
    data_processor = DataProcessor(cfg)
    processed_data = data_processor.process(data)
    
    # Create datasets for AE
    train_dataset_ae, val_dataset_ae, test_dataset_ae = data_processor.create_ae_datasets(
        processed_data
    )

    # Create data loaders
    train_loader_ae = DataLoader(
        train_dataset_ae, 
        batch_size=cfg.model.autoencoder.batch_size, 
        shuffle=True,
        num_workers=cfg.data.loader.num_workers,
        pin_memory=cfg.data.loader.pin_memory
    )
    val_loader_ae = DataLoader(
        val_dataset_ae, 
        batch_size=cfg.model.autoencoder.batch_size, 
        shuffle=False,
        num_workers=cfg.data.loader.num_workers,
        pin_memory=cfg.data.loader.pin_memory
    )

    # 3. Train or load Autoencoder——————————————————————————————
    autoencoder = AutoencoderModel(cfg).to(device)
    
    if cfg.model.ae_lstm.use_pretrained_ae:
        # Load pretrained autoencoder
        ae_path = os.path.join(cfg.paths.ae_model_dir, "best_model.pt")
        if os.path.exists(ae_path):
            autoencoder.load_state_dict(torch.load(ae_path, map_location=device))
            log.info(f"Loaded pretrained autoencoder from {ae_path}")
        else:
            log.warning(f"No pretrained autoencoder found at {ae_path}. Training new model.")
            train_autoencoder(autoencoder, train_loader_ae, val_loader_ae, cfg, device, writer)
    else:
        # Train autoencoder from scratch
        train_autoencoder(autoencoder, train_loader_ae, val_loader_ae, cfg, device, writer)
    
    # Generate latent representations for LSTM training
    log.info("Generating latent representations for LSTM training")
    train_latent, val_latent, test_latent = data_processor.generate_latent_representations(
        autoencoder, train_dataset_ae, val_dataset_ae, test_dataset_ae, device
    )
    
    # Create sequence datasets for LSTM
    train_sequence_lstm, val_sequence_lstm, test_sequence_lstm = data_processor.create_sequence_datasets(
        train_latent, val_latent, test_latent, processed_data
    )

    batch_size = cfg.model.lstm.batch_size

    train_loader = DataLoader(
        train_sequence_lstm, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=cfg.data.loader.num_workers,
        pin_memory=cfg.data.loader.pin_memory
    )
    val_loader = DataLoader(
        val_sequence_lstm, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=cfg.data.loader.num_workers,
        pin_memory=cfg.data.loader.pin_memory
    )
    test_loader = DataLoader(
        test_sequence_lstm, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=cfg.data.loader.num_workers,
        pin_memory=cfg.data.loader.pin_memory
    )


    # 4. Train or load LSTM——————————————————————————————————————————————————————————
    cfg.model.lstm.input_size = cfg.model.autoencoder.latent_dim
    cfg.model.lstm.output_size = cfg.model.autoencoder.latent_dim
    lstm_model = LSTMModel(cfg).to(device)
    
    if cfg.model.ae_lstm.use_pretrained_lstm:
        # Load pretrained LSTM
        lstm_path = os.path.join(cfg.paths.lstm_model_dir, "best_model.pt")
        if os.path.exists(lstm_path):
            lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
            log.info(f"Loaded pretrained LSTM from {lstm_path}")
    else:
        train_lstm(lstm_model, train_loader, val_loader, cfg, device, writer)
    
    # 5. Fine-tune AE-LSTM model (optional fine-tuning)——————————————————————————
    ae_lstm_model = AELSTMModel(autoencoder, lstm_model, cfg).to(device)
    
    if cfg.model.ae_lstm.train_ae_end_to_end:
        log.info("Fine-tuning the combined AE-LSTM model")
        train_ae_lstm(ae_lstm_model, train_dataset_ae, val_dataset_ae, cfg, device, writer)
    
    # 6. Evaluation————————————————————————————
    log.info("Evaluating models")
    log_memory_usage("Before evaluation")
    
    # Force garbage collection before evaluation
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if cfg.evaluation.use_test_set:
        # Evaluate LSTM prediction
        lstm_metrics = evaluate_model(
            lstm_model,
            test_loader,
            test_dataset_ae.tensors[0].cpu().numpy(),
            cfg,
            device,
            autoencoder_model=autoencoder,
            model_type="lstm"
        )
        log_memory_usage("After test evaluation")
    if cfg.evaluation.use_val_set:
        # Evaluate LSTM prediction
        lstm_metrics = evaluate_model(
            lstm_model,
            val_loader,
            val_dataset_ae.tensors[0].cpu().numpy(),
            cfg,
            device,
            autoencoder_model=autoencoder,
            model_type="lstm"
        )       


    # Close TensorBoard writer
    if writer:
        writer.close()
    
    log.info("Pipeline completed successfully!")
    
    final_metric = lstm_metrics['metrics'].get('rmse', float('inf')) # Default to infinity if RMSE not found
    log.info(f"Returning metric for Optuna: {final_metric}")
    return final_metric


if __name__ == "__main__":
    main()
