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
# from src.model.lstm import LSTMModel # Replaced by factory
from src.model.factory import get_model # Added model factory
from src.model.ae_predictive import AEPredictiveModel # Updated import
from src.train.train_autoencoder import train_autoencoder
from src.train.train_model import train_model # Updated import
from src.train.train_ae_predictive import train_ae_predictive # Updated import
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
        batch_size=cfg.train.autoencoder.batch_size, 
        shuffle=True,
        num_workers=cfg.data.loader.num_workers,
        pin_memory=cfg.data.loader.pin_memory
    )
    val_loader_ae = DataLoader(
        val_dataset_ae, 
        batch_size=cfg.train.autoencoder.batch_size, 
        shuffle=False,
        num_workers=cfg.data.loader.num_workers,
        pin_memory=cfg.data.loader.pin_memory
    )

    # 3. Train or load Autoencoder——————————————————————————————
    autoencoder = AutoencoderModel(cfg).to(device)
    
    train_autoencoder(autoencoder, train_loader_ae, val_loader_ae, cfg, device, writer)

    active_model_name = cfg.model.name.lower()
    # Generate latent representations for predictive model training
    log.info(f"Generating latent representations for {active_model_name.upper()} training") # Generalized log
    train_latent, val_latent, test_latent = data_processor.generate_latent_representations(
        autoencoder, train_dataset_ae, val_dataset_ae, test_dataset_ae, device
    )
    
    # Create sequence datasets for the predictive model
    train_sequence_predictive, val_sequence_predictive, test_sequence_predictive = data_processor.create_sequence_datasets( # Renamed variables
        train_latent, val_latent, test_latent, processed_data
    )

    # Get batch_size from the active predictive model's training configuration
    # This assumes batch_size is defined under cfg.train[active_model_name].batch_size
    active_train_cfg = cfg.train[active_model_name]
    active_model_cfg = cfg.model[active_model_name]
    predictive_model_batch_size = active_train_cfg.batch_size

    train_loader = DataLoader(
        train_sequence_predictive, # Use renamed variable
        batch_size=predictive_model_batch_size, # Use batch_size from active model's train config
        shuffle=True,
        num_workers=cfg.data.loader.num_workers,
        pin_memory=cfg.data.loader.pin_memory
    )
    val_loader = DataLoader(
        val_sequence_predictive, # Use renamed variable
        batch_size=predictive_model_batch_size, # Use batch_size from active model's train config
        shuffle=False,
        num_workers=cfg.data.loader.num_workers,
        pin_memory=cfg.data.loader.pin_memory
    )
    test_loader = DataLoader(
        test_sequence_predictive, # Use renamed variable
        batch_size=predictive_model_batch_size, # Use batch_size from active model's train config
        shuffle=False,
        num_workers=cfg.data.loader.num_workers,
        pin_memory=cfg.data.loader.pin_memory
    )


    # 4. Train or load Predictive Model (LSTM, Mamba, Transformer)————————————————————
    log.info(f"Active model type: {active_model_name}")

    # Dynamically set input_size and output_size for the chosen predictive model
    # These keys (input_size, output_size) are expected to be directly under cfg.model
    # as they are loaded from the selected model_configs/*.yaml file.
    active_model_cfg.input_size = cfg.model.autoencoder.latent_dim
    active_model_cfg.output_size = cfg.model.autoencoder.latent_dim

    # Log the updated sizes for the active model
    log.info(f"Setting input_size={active_model_cfg.input_size}, output_size={active_model_cfg.output_size} for {active_model_name} model based on AE latent_dim.")

    model = get_model(cfg).to(device) # Instantiating the model using the factory
    
    # Determine model directory dynamically
    # This assumes paths like 'lstm_model_dir', 'mamba_model_dir' are in cfg.paths
    model_specific_dir_name = f"{active_model_name}_model_dir"
    # Default path if specific one is not found in config (though it should be added)
    default_model_path = os.path.join(cfg.paths.output_dir, "models", active_model_name) # Should be cfg.paths.model_dir
    model_dir = cfg.paths.get(model_specific_dir_name, default_model_path)

    # Use the generalized config key: cfg.model.ae_predictive.use_pretrained_predictive_model
    # This flag indicates if the predictive model component of the AEPredictiveModel should be loaded.
    # The main `model` is already loaded or trained by this point.
    # This logic seems to be for the predictive model itself, not the combined one.
    if cfg.model.ae_predictive.use_pretrained_predictive_model:
        model_path = os.path.join(model_dir, "best_model.pt") # model_dir is already predictive model specific
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            log.info(f"Loaded pretrained {active_model_name} model from {model_path}")
        else:
            log.warning(f"No pretrained {active_model_name} model found at {model_path}. Training new model.")
            # Assuming train_model can be generalized or selected based on model type
            # For now, this might only work correctly if the new models are trained with train_model
            # or if train_model is refactored into a generic train_model function.
            train_model(model, train_loader, val_loader, cfg, device, writer) # Updated function call
    else:
        # Same assumption as above about train_model
        train_model(model, train_loader, val_loader, cfg, device, writer) # Updated function call
    
    # 5. Fine-tune AE-Predictive_Model (optional fine-tuning)——————————————————————————
    # The name in the config cfg.model.ae_lstm (soon to be ae_predictive) should ideally also be generalized.
    ae_predictive_model = AEPredictiveModel(autoencoder, model, cfg).to(device) # Updated class name
    
    if cfg.model.ae_predictive.train_ae_end_to_end: # Updated to use ae_predictive
        log.info(f"Fine-tuning the combined AE-{active_model_name.upper()} model")
        # train_ae_predictive (formerly train_ae_lstm) might also need to be generalized.
        # It also uses cfg.train.ae_predictive (formerly ae_lstm).
        # train_ae_predictive.py now reads from cfg.train.ae_predictive.
        train_ae_predictive(ae_predictive_model, train_dataset_ae, val_dataset_ae, cfg, device, writer) # Updated call
    
    # 6. Evaluation————————————————————————————
    log.info(f"Evaluating {active_model_name.upper()} model")
    log_memory_usage("Before evaluation")
    
    # Force garbage collection before evaluation
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if cfg.evaluation.use_test_set:
        # Evaluate predictive model
        model_metrics = evaluate_model( # Renamed lstm_metrics to model_metrics
            model, # Pass the generic model
            test_loader,
            test_dataset_ae.tensors[0].cpu().numpy(), # Assuming this is the original scale data for error calculation
            cfg,
            device,
            autoencoder_model=autoencoder,
            model_type=active_model_name # Pass active model name
        )
        log_memory_usage("After test evaluation")
    if cfg.evaluation.use_val_set:
        # Evaluate predictive model
        model_metrics = evaluate_model( # Renamed lstm_metrics to model_metrics
            model, # Pass the generic model
            val_loader,
            val_dataset_ae.tensors[0].cpu().numpy(), # Assuming this is the original scale data
            cfg,
            device,
            autoencoder_model=autoencoder,
            model_type=active_model_name # Pass active model name
        )       


    # Close TensorBoard writer
    if writer:
        writer.close()
    
    log.info("Pipeline completed successfully!")
    
    # Ensure model_metrics is defined, e.g. from the last evaluation run (val or test)
    # If only test_set is true, model_metrics from val_set block won't be defined.
    # Defaulting to an empty dict if no evaluation was run, though this case should be handled.
    final_metric_value = model_metrics['metrics'].get('rmse', float('inf')) if 'model_metrics' in locals() and model_metrics else float('inf')
    log.info(f"Returning metric for Optuna: {final_metric_value}")
    return final_metric_value


if __name__ == "__main__":
    main()
