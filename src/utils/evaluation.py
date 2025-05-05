#!/usr/bin/env python
# src/utils/evaluation.py - Model evaluation utilities

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from omegaconf import DictConfig

log = logging.getLogger(__name__)

def calculate_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate evaluation metrics between true and predicted values.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        mask: Optional mask to apply (1 for valid data points, 0 for invalid)
    
    Returns:
        Dictionary of metrics
    """
    if mask is not None:
        # Apply mask
        y_true = y_true * mask
        y_pred = y_pred * mask
        
        # Count valid points for averaging
        valid_points = mask.sum()
    else:
        valid_points = y_true.size
    
    # Calculate error
    error = y_pred - y_true
    abs_error = np.abs(error)
    squared_error = error ** 2
    
    # Calculate metrics
    mae = abs_error.sum() / valid_points
    rmse = np.sqrt(squared_error.sum() / valid_points)
    
    # Calculate R^2
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum(squared_error)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Calculate MAPE where y_true != 0
    nonzero_mask = (np.abs(y_true) > 1e-6)
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs(error[nonzero_mask] / y_true[nonzero_mask])) * 100
    else:
        mape = float('nan')
    
    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape)
    }
    
    return metrics


def evaluate_model(
    model: nn.Module,
    dataset: Dataset,
    cfg: DictConfig,
    device: torch.device,
    model_type: str = "autoencoder",
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Evaluate a model on the given dataset.
    
    Args:
        model: PyTorch model to evaluate
        dataset: Dataset for evaluation
        cfg: Configuration object
        device: Device to run evaluation on
        model_type: Type of model to evaluate ("autoencoder", "lstm", or "ae_lstm")
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary of evaluation metrics and predictions
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_inputs = []
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            if model_type == "autoencoder":
                inputs = batch[0].to(device)
                targets = inputs  # For autoencoders, targets = inputs
                outputs = model(inputs)
            elif model_type == "lstm":
                inputs, targets = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
            elif model_type == "ae_lstm":
                inputs, targets = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            all_inputs.append(inputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())
    
    # Concatenate results
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    # Load mask if specified
    mask = None
    if cfg.data.preprocessing.apply_mask:
        try:
            mask = np.load(cfg.data.preprocessing.mask_path)
        except Exception as e:
            log.warning(f"Could not load mask file: {e}")
    
    # Calculate metrics
    metrics = calculate_metrics(all_targets, all_predictions, mask)
    
    log.info(f"Evaluation metrics for {model_type}:")
    for metric_name, metric_value in metrics.items():
        log.info(f"  {metric_name}: {metric_value:.4f}")
    
    # Return results
    return {
        "metrics": metrics,
        "inputs": all_inputs,
        "targets": all_targets,
        "predictions": all_predictions
    }


def calculate_spatial_metrics(
    predictions: np.ndarray, 
    targets: np.ndarray, 
    mask: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Calculate metrics for each spatial location over time.
    
    Args:
        predictions: Predicted values with shape [time, height, width]
        targets: Target values with shape [time, height, width]
        mask: Optional spatial mask with shape [height, width]
    
    Returns:
        Dictionary of spatial metrics arrays
    """
    if mask is not None:
        # Expand mask to match time dimension
        expanded_mask = np.expand_dims(mask, axis=0)
        expanded_mask = np.repeat(expanded_mask, predictions.shape[0], axis=0)
        
        # Apply mask
        predictions = predictions * expanded_mask
        targets = targets * expanded_mask
    
    # Initialize metric arrays
    height, width = targets.shape[1], targets.shape[2]
    mae_map = np.zeros((height, width))
    rmse_map = np.zeros((height, width))
    r2_map = np.zeros((height, width))
    
    # Calculate metrics for each spatial location
    for i in range(height):
        for j in range(width):
            # Skip if masked out
            if mask is not None and mask[i, j] == 0:
                continue
                
            # Extract time series for this location
            y_true = targets[:, i, j]
            y_pred = predictions[:, i, j]
            
            # Calculate metrics
            error = y_pred - y_true
            mae_map[i, j] = np.mean(np.abs(error))
            rmse_map[i, j] = np.sqrt(np.mean(error ** 2))
            
            # Calculate R^2
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            ss_res = np.sum(error ** 2)
            r2_map[i, j] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        "mae_map": mae_map,
        "rmse_map": rmse_map,
        "r2_map": r2_map
    }
