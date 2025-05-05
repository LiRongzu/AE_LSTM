#!/usr/bin/env python
# src/utils/setup.py - Experiment setup utilities

import os
import random
import logging
import numpy as np
import torch
import json
from omegaconf import DictConfig, OmegaConf

def setup_experiment(cfg: DictConfig) -> None:
    """
    Setup the experiment by creating necessary directories.
    
    Args:
        cfg: Hydra configuration
    """
    # Create output directories
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    os.makedirs(cfg.paths.model_dir, exist_ok=True)
    os.makedirs(cfg.paths.ae_model_dir, exist_ok=True)
    os.makedirs(cfg.paths.lstm_model_dir, exist_ok=True)
    os.makedirs(cfg.paths.combined_model_dir, exist_ok=True)
    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    os.makedirs(cfg.paths.visualization_dir, exist_ok=True)
    os.makedirs(cfg.paths.logs_dir, exist_ok=True)
    
    if cfg.logging.use_tensorboard:
        os.makedirs(cfg.paths.tensorboard_dir, exist_ok=True)


def setup_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(cfg: DictConfig) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        Logger instance
    """
    # Hydra configures basic logging automatically, 
    # but we can add extra handlers or formatters if needed
    logger = logging.getLogger()
    log_level = getattr(logging, cfg.logging.level.upper())
    logger.setLevel(log_level)
    
    # Add file handler if not already added by Hydra
    if cfg.logging.save_dir:
        os.makedirs(cfg.logging.save_dir, exist_ok=True)
        log_file = os.path.join(cfg.logging.save_dir, f"{cfg.experiment.name}.log")
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


