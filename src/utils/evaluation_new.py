#!/usr/bin/env python
# src/utils/evaluation.py - Model evaluation utilities

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
from typing import Dict, Any, List, Optional, Tuple
from omegaconf import DictConfig
import joblib
import os

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
    
    # Ensure y_true and y_pred are 2D for metric calculations if they are not already
    if y_true.ndim > 2:
        y_true = y_true.reshape(-1, y_true.shape[-1])
    if y_pred.ndim > 2:
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])

    if y_true.shape != y_pred.shape:
        log.error(f"Shape mismatch in calculate_metrics. y_true: {y_true.shape}, y_pred: {y_pred.shape}")
        return {
            'mae': np.nan, 'rmse': np.nan, 'r2': np.nan, 'mape': np.nan
        }
    
    return metrics


def model_inference(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    model_type: str = "lstm",
    autoencoder_model: Optional[torch.nn.Module] = None,
    use_sliding_window: bool = False,
    prediction_steps: int = 1,
    device: torch.device = None
) -> np.ndarray:
    """
    Perform model inference with options for single-step or multi-step predictions.
    
    Args:
        model: PyTorch model to use for inference (LSTM or AE-LSTM)
        inputs: Input tensor for prediction
        model_type: Type of model ("lstm" or "ae_lstm")
        autoencoder_model: Optional autoencoder model for decoding LSTM outputs
        use_sliding_window: Whether to use sliding window for multi-step prediction
        prediction_steps: Number of steps to predict in multi-step mode
        device: Device to run inference on
    
    Returns:
        Numpy array of predictions
    """
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    inputs = inputs.to(device)
    
    if not use_sliding_window:
        # Single-step prediction
        with torch.no_grad():
            if model_type == "lstm":
                # LSTM outputs latent representations
                latent_outputs = model(inputs)
                
                # Decode using autoencoder if provided
                if autoencoder_model is None:
                    raise ValueError("Autoencoder model must be provided for LSTM inference")
                
                # Handle different output shapes
                if latent_outputs.ndim == 3 and latent_outputs.shape[1] == 1:
                    latent_outputs = latent_outputs.squeeze(1)
                
                outputs = autoencoder_model.decode(latent_outputs)
            
            else:  # ae_lstm
                # AE-LSTM directly outputs in the original feature space
                outputs = model(inputs)
            
            return outputs.cpu().numpy()
    
    else:
        # Sliding window multi-step iterative prediction
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        feature_dim = inputs.shape[2]
        
        # Initialize storage for multi-step predictions
        multi_step_preds = np.zeros((batch_size, prediction_steps, feature_dim))
        
        for sample_idx in range(batch_size):
            # Get single sample
            sample_input = inputs[sample_idx:sample_idx+1]  # Keep batch dimension
            
            # Initialize rolling input
            rolling_input = sample_input.clone()
            
            for step in range(prediction_steps):
                with torch.no_grad():
                    if model_type == "lstm":
                        # LSTM prediction
                        latent_pred = model(rolling_input)
                        
                        # Decode prediction
                        if latent_pred.ndim == 3 and latent_pred.shape[1] == 1:
                            latent_pred = latent_pred.squeeze(1)
                        
                        pred = autoencoder_model.decode(latent_pred)
                    else:  # ae_lstm
                        # AE-LSTM prediction
                        pred = model(rolling_input)
                
                # Store prediction
                multi_step_preds[sample_idx, step] = pred.cpu().numpy()
                
                # Update rolling input for next step
                if rolling_input.shape[1] > 1:
                    # Shift window and add new prediction at the end
                    rolling_input = torch.cat([
                        rolling_input[:, 1:], 
                        pred.unsqueeze(1)
                    ], dim=1)
                else:
                    # If sequence length is 1, just replace with new prediction
                    rolling_input = pred.unsqueeze(1)
        
        return multi_step_preds


def predict_with_dataloader(
    model: torch.nn.Module,
    data_loader: DataLoader,
    model_type: str = "lstm",
    autoencoder_model: Optional[torch.nn.Module] = None,
    use_sliding_window: bool = False,
    prediction_steps: int = 1,
    device: torch.device = None,
    scaler = None
) -> np.ndarray:
    """
    Perform model inference on a DataLoader with options for single-step or multi-step predictions.
    
    Args:
        model: PyTorch model to use for inference (LSTM or AE-LSTM)
        data_loader: DataLoader containing input data
        model_type: Type of model ("lstm" or "ae_lstm")
        autoencoder_model: Optional autoencoder model for decoding LSTM outputs
        use_sliding_window: Whether to use sliding window for multi-step prediction
        prediction_steps: Number of steps to predict in multi-step mode
        device: Device to run inference on
        scaler: Optional scaler for inverse normalization
    
    Returns:
        Numpy array of predictions for all batches
    """
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    all_predictions = []
    
    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc=f"Predicting with {model_type}"):
            inputs = batch_data[0].to(device)
            
            # Use the model_inference function for each batch
            batch_predictions = model_inference(
                model=model,
                inputs=inputs,
                model_type=model_type,
                autoencoder_model=autoencoder_model,
                use_sliding_window=use_sliding_window,
                prediction_steps=prediction_steps,
                device=device
            )
            
            # Apply inverse normalization if scaler is provided
            if scaler is not None:
                if use_sliding_window:
                    # For multi-step predictions
                    feature_dim = batch_predictions.shape[-1]
                    batch_predictions = scaler.inverse_transform(
                        batch_predictions.reshape(-1, feature_dim)
                    ).reshape(batch_predictions.shape)
                else:
                    # For single-step predictions
                    batch_predictions = scaler.inverse_transform(
                        batch_predictions.reshape(-1, batch_predictions.shape[-1])
                    ).reshape(batch_predictions.shape)
            
            all_predictions.append(batch_predictions)
    
    # Concatenate all batch predictions
    return np.concatenate(all_predictions, axis=0)


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    y_true: np.ndarray,
    cfg: DictConfig,
    device: torch.device,
    model_type: str = "lstm",
    autoencoder_model: Optional[torch.nn.Module] = None,
    use_sliding_window: bool = False,
    prediction_steps: int = 1,
    scaler = None
) -> Dict[str, Any]:
    """
    Evaluate a model on the given dataset with options for single-step or multi-step predictions.
    
    Args:
        model: PyTorch model to evaluate (LSTM or AE-LSTM)
        test_loader: DataLoader for evaluation
        y_true: Ground truth values
        cfg: Configuration object
        device: Device to run evaluation on
        model_type: Type of model to evaluate ("lstm" or "ae_lstm")
        autoencoder_model: Optional autoencoder model for decoding LSTM outputs
        use_sliding_window: Whether to use sliding window for multi-step prediction
        prediction_steps: Number of steps to predict in multi-step mode
        scaler: Scaler object for inverse normalization
    
    Returns:
        Dictionary of evaluation metrics and predictions
    """
    model.eval()
    
    # Load scaler if path is provided
    if scaler is None:
        scaler = cfg.paths.scaler_path
    
    if isinstance(scaler, str):
        if os.path.exists(scaler):
            try:
                scaler = joblib.load(scaler)
                log.info(f"Loaded scaler from path: {scaler}")
            except Exception as e:
                log.error(f"Failed to load scaler from path {scaler}: {e}. Proceeding without scaler.")
                scaler = None
        else:
            log.warning(f"Scaler path {scaler} not found. Proceeding without scaler.")
            scaler = None

    log.info(f"Starting evaluation for model_type: {model_type}")
    log.info(f"Evaluation mode: {'sliding window multi-step' if use_sliding_window else 'single-step'}")
    log.info(f"Test loader size: {len(test_loader)}")

    # Use the new predict_with_dataloader function
    all_predictions = predict_with_dataloader(
        model=model,
        data_loader=test_loader,
        model_type=model_type,
        autoencoder_model=autoencoder_model,
        use_sliding_window=use_sliding_window,
        prediction_steps=prediction_steps,
        device=device,
        scaler=scaler
    )
    
    log.info(f"Final shapes - Predictions: {all_predictions.shape}")

    # Calculate metrics
    y_true = y_true[-all_predictions.shape[0]:]
    
    # Apply inverse normalization to y_true if scaler is provided
    if scaler is not None:
        y_true = scaler.inverse_transform(y_true.reshape(-1, y_true.shape[-1])).reshape(y_true.shape)

    metrics = calculate_metrics(y_true, all_predictions)

    log.info("Metrics calculated:")
    for metric_name, value in metrics.items():
        log.info(f"  {metric_name}: {value:.4f}")
        
    return {
        "metrics": metrics,
        "predictions": all_predictions,
    }


def evaluate(cfg,
             model: torch.nn.Module,
             test_loader: DataLoader,
             y_true: np.ndarray,
             device: torch.device,
             model_type: str = "lstm",
             autoencoder_model: Optional[torch.nn.Module] = None,
             use_sliding_window: bool = False,
             prediction_steps: int = 1,
             scaler = None) -> Dict[str, Any]:
    """
    Wrapper for evaluate_model to handle configuration and logging.
    
    Args:
        cfg: Configuration object
        model: PyTorch model to evaluate
        test_loader: DataLoader for evaluation
        y_true: Ground truth values
        device: Device to run evaluation on
        model_type: Type of model to evaluate ("lstm" or "ae_lstm")
        autoencoder_model: Optional autoencoder model for decoding LSTM outputs
        use_sliding_window: Whether to use sliding window for multi-step prediction
        prediction_steps: Number of steps to predict in multi-step mode
        scaler: Scaler object for inverse normalization
    
    Returns:
        Dictionary of evaluation metrics and predictions
    """
    log.info(f"Starting evaluation for {model_type} model")
    
    # Call the actual evaluation function
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        y_true=y_true,
        cfg=cfg,
        device=device,
        model_type=model_type,
        autoencoder_model=autoencoder_model,
        use_sliding_window=use_sliding_window,
        prediction_steps=prediction_steps,
        scaler=scaler
    )
    
    return metrics
