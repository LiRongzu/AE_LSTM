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
    model_type: str = "lstm", # model_type is still useful for logging or other specific logic
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
            # If autoencoder_model is provided, 'model' is a predictive model (LSTM, Mamba, Transformer)
            # whose outputs (latent codes) need decoding.
            # If autoencoder_model is None, 'model' is an end-to-end model (e.g., AEPredictiveModel)
            # that directly outputs in the original feature space.
            if autoencoder_model is not None:
                latent_outputs = model(inputs) # Output from LSTM, Mamba, or Transformer
                
                # Handle different output shapes from predictive models if necessary
                # (e.g. if they output full sequence vs. last step)
                # Assuming predictive models (LSTM, Mamba, Transformer) called here are configured
                # to output (batch, latent_dim) as per their design in previous steps.
                if latent_outputs.ndim == 3 and latent_outputs.shape[1] == 1: # (B, 1, D) -> (B, D)
                    latent_outputs = latent_outputs.squeeze(1)
                elif latent_outputs.ndim == 3 and latent_outputs.shape[1] > 1:
                    # This case should ideally not happen if predictive models are designed
                    # to output only the last step for this type of evaluation.
                    # If they do output sequences, taking the last step here.
                    log.warning(f"model_inference (single-step): Predictive model outputted sequence (shape {latent_outputs.shape}), taking last time step.")
                    latent_outputs = latent_outputs[:, -1, :]
                
                outputs = autoencoder_model.decode(latent_outputs)
            else:  # autoencoder_model is None, so 'model' is an end-to-end model (e.g. AEPredictiveModel)
                outputs = model(inputs)
            
            return outputs.cpu().numpy()
    
    else:
        # Sliding window multi-step iterative prediction
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        input_feature_dim = inputs.shape[2]  # This is latent dimension for LSTM
        
        # Determine output feature dimension based on whether AE is used
        if autoencoder_model is not None: # Predictive model (LSTM, Mamba, etc.) + AE
            if not hasattr(autoencoder_model, 'input_dim'):
                 # A basic check; ideally autoencoder_model would have a clear property for original input dim.
                 # This might happen if a non-standard AE is passed.
                raise ValueError("autoencoder_model provided but lacks 'input_dim' attribute for determining output_feature_dim.")
            output_feature_dim = autoencoder_model.input_dim
        else: # End-to-end model (AEPredictiveModel)
              # input_feature_dim for AEPredictiveModel would be the original data's feature dim.
              # However, the 'inputs' to this function for AEPredictiveModel in sliding window
              # would be actual data, not latent codes.
              # This part needs careful thought for AEPredictiveModel in sliding window.
              # The current AEPredictiveModel.predict_sequence is very LSTM-specific.
              # For now, let's assume if autoencoder_model is None, it's an AEPredictiveModel,
              # and its output_size (if it has one) or input_size should be the feature dim.
              # This is tricky because AEPredictiveModel's direct input IS the original feature dim.
            if hasattr(model, 'autoencoder') and hasattr(model.autoencoder, 'input_dim'): # If it's AEPredictiveModel
                 output_feature_dim = model.autoencoder.input_dim
            elif hasattr(model, 'output_size'): # Fallback if it's some other end-to-end model
                 output_feature_dim = model.output_size
            else: # Fallback to input's feature dim if model has no clear output_size attribute
                 output_feature_dim = inputs.shape[-1]
                 log.warning(f"Could not reliably determine output_feature_dim for sliding window with model type {model_type} and no autoencoder. Defaulting to input feature dim: {output_feature_dim}")


        # Initialize storage for multi-step predictions
        multi_step_preds = np.zeros((batch_size, prediction_steps, output_feature_dim))
        
        for sample_idx in range(batch_size):
            # Get single sample
            sample_input = inputs[sample_idx:sample_idx+1]  # Keep batch dimension
            
            # Initialize rolling input
            rolling_input = sample_input.clone()
            
            for step in range(prediction_steps):
                with torch.no_grad():
                    if autoencoder_model is not None: # Predictive model (LSTM, Mamba, etc.) + AE
                        latent_pred = model(rolling_input) # rolling_input is latent codes here
                        
                        if latent_pred.ndim == 3 and latent_pred.shape[1] == 1:
                            latent_pred = latent_pred.squeeze(1)
                        elif latent_pred.ndim == 3 and latent_pred.shape[1] > 1:
                             log.warning(f"model_inference (sliding_window): Predictive model outputted sequence (shape {latent_pred.shape}), taking last time step.")
                             latent_pred = latent_pred[:, -1, :]
                        
                        pred = autoencoder_model.decode(latent_pred)
                        
                        # For rolling input update, use latent prediction (not decoded)
                        next_input_for_roll = latent_pred # This should be (B, latent_dim)
                    else:  # End-to-end model (AEPredictiveModel)
                           # rolling_input is in original data space here.
                        pred = model(rolling_input) # pred is in original data space
                        
                        # For rolling input update, AE-Predictive model needs latent codes.
                        # This is complex: to feed back, we'd need to encode `pred` if the AEPredictiveModel's
                        # internal predictive model expects latent codes.
                        # The AEPredictiveModel.predict_sequence handles this iteration internally.
                        # This external sliding window loop is more for a standalone predictive model + AE.
                        # If 'model' is AEPredictiveModel, its own .predict_sequence should be called,
                        # not this external loop.
                        # This highlights a potential design issue if trying to use this generic
                        # sliding window for an end-to-end model that has its own sequence handling.
                        # For now, assuming if autoencoder_model is None, this path might be problematic
                        # if the model isn't simple stateleess or if its predict_sequence isn't used.
                        # Let's assume 'pred' is what should be fed back if no AE.
                        log.warning("Sliding window for end-to-end model without explicit AE for feedback encoding might be inaccurate if model expects latent codes internally for sequence prediction.")
                        next_input_for_roll = pred # This might be wrong if model expects latent codes.

                # Store prediction
                multi_step_preds[sample_idx, step] = pred.cpu().numpy() # pred is in original data space
                
                # Update rolling input for next step
                # next_input_for_roll should be (B, feature_dim_for_predictive_model_input)
                # If AE is used, feature_dim is latent_dim. If no AE, feature_dim is original_dim.
                if rolling_input.shape[1] > 1:
                    rolling_input = torch.cat([
                        rolling_input[:, 1:], 
                        next_input_for_roll.unsqueeze(1) # Ensure (B, 1, D)
                    ], dim=1)
                else:
                    rolling_input = next_input_for_roll.unsqueeze(1)
        
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
    
    # Memory optimization: import garbage collection
    import gc
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(data_loader, desc=f"Predicting with {model_type}")):
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
            
            # Memory cleanup after each batch
            del inputs, batch_predictions
            if batch_idx % 1 == 0:  # Clean up every batch for this small dataset
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    # Concatenate all batch predictions
    return np.concatenate(all_predictions, axis=0)

def data_inverse(cfg, data):
    scaler = cfg.paths.scaler_path 
    if isinstance(scaler, str): # 如果 scaler 是路径字符串
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

    data = scaler.inverse_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
    return data


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    y_true: np.ndarray,
    cfg: DictConfig,
    device: torch.device,
    model_type: str = "lstm",
    autoencoder_model: Optional[torch.nn.Module] = None,
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
    
    Returns:
        Dictionary of evaluation metrics and predictions
    """
    model.eval()
    log.info(f"Starting evaluation for model_type: {model_type}")
    log.info(f"Test loader size: {len(test_loader)}")

    # Use the new predict_with_dataloader function
    # Memory-efficient prediction with garbage collection
    import gc
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    all_predictions = predict_with_dataloader(
        model=model,
        data_loader=test_loader,
        model_type=model_type,
        autoencoder_model=autoencoder_model,
        use_sliding_window=cfg.evaluation.use_sliding_window,
        prediction_steps=cfg.evaluation.prediction_steps if cfg.evaluation.use_sliding_window else 1,
        device=device,
        scaler=scaler
    )
    
    log.info(f"Final shapes - Predictions: {all_predictions.shape}")
    
    # Force garbage collection after predictions
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Calculate metrics with memory optimization
    y_true = data_inverse(cfg, y_true)
    y_true = y_true[-all_predictions.shape[0]:]
    
    # Force garbage collection before metrics calculation
    gc.collect()
    
    metrics = calculate_metrics(y_true, all_predictions)
    
    # Clear intermediate variables
    del y_true
    gc.collect()

    log.info("Metrics calculated:")
    for metric_name, value in metrics.items():
        log.info(f"  {metric_name}: {value:.4f}")
        
    return {
        "metrics": metrics,
        "predictions": all_predictions,
    }
