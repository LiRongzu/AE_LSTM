#!/usr/bin/env python
# src/train/train_lstm.py - LSTM training module

import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple, Optional, Union, Any
from omegaconf import DictConfig
from tqdm import tqdm # Import tqdm
from src.train.train_autoencoder import EarlyStopping  # Reuse the EarlyStopping class

log = logging.getLogger(__name__)

def train_model( # Renamed from train_lstm
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: DictConfig,
    device: torch.device,
    writer: Optional[SummaryWriter] = None
) -> nn.Module:
    """
    Train a predictive model (LSTM, Mamba, Transformer).

    Args:
        model: Model to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        cfg: Configuration object
        device: Device to train on
        writer: Optional TensorBoard writer

    Returns:
        Trained model
    """
    model_name = cfg.model.name.lower()
    # model_specific_cfg (architectural) is now directly cfg.model due to Hydra merge
    # train_specific_cfg is for training process parameters for this model
    train_specific_cfg = cfg.train[model_name]

    # Extract training parameters from train_specific_cfg
    epochs = train_specific_cfg.epochs
    learning_rate = train_specific_cfg.learning_rate
    weight_decay = train_specific_cfg.weight_decay
    # Batch size for DataLoader is usually defined when creating DataLoader in main_pipeline
    # However, if it's needed here for some reason, it should also come from train_specific_cfg
    # batch_size = train_specific_cfg.batch_size

    log.info(f"Training {model_name} with LR: {learning_rate}, Weight Decay: {weight_decay}, Epochs: {epochs}")

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Create learning rate scheduler
    scheduler_cfg = train_specific_cfg.scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=scheduler_cfg.factor,
        patience=scheduler_cfg.patience,
        min_lr=scheduler_cfg.min_lr
        # verbose=True # Removed, not a valid argument for ReduceLROnPlateau in current PyTorch
    )

    # Loss function (MSE for prediction)
    criterion = nn.MSELoss()

    # Early stopping
    early_stopping_cfg = train_specific_cfg.early_stopping
    early_stopping = EarlyStopping(
        patience=early_stopping_cfg.patience,
        min_delta=early_stopping_cfg.min_delta,
        mode='min'
    )

    # Determine model directory dynamically
    model_save_dir = cfg.paths.get(f"{model_name}_model_dir", os.path.join(cfg.paths.model_dir, model_name))
    os.makedirs(model_save_dir, exist_ok=True)

    # Training loop
    log.info(f"Beginning {model_name.upper()} training for {epochs} epochs. Saving models to {model_save_dir}")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        # Training
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False, unit="batch")
        for batch_idx, (inputs, targets) in enumerate(train_progress_bar):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update statistics
            epoch_loss += loss.item()

            # Update tqdm postfix with current batch loss
            train_progress_bar.set_postfix(loss=f"{loss.item():.6f}")

            # Log batch loss to TensorBoard (optional, for very detailed tracking)
            if writer and cfg.logging.log_batch_metrics: # Add a config for this if desired
                current_iter = epoch * len(train_loader) + batch_idx
                writer.add_scalar(f'{model_name}/Loss/train_batch', loss.item(), current_iter)

        # Calculate average epoch loss
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_predictions = []

        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False, unit="batch")
        with torch.no_grad():
            for val_inputs, val_targets in val_progress_bar:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs = model(val_inputs)
                val_batch_loss = criterion(val_outputs, val_targets)
                val_loss += val_batch_loss.item()
                val_progress_bar.set_postfix(loss=f"{val_batch_loss.item():.6f}")

                # Collect predictions and targets for metrics
                all_targets.append(val_targets.cpu().numpy())
                all_predictions.append(val_outputs.cpu().numpy())

        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Log to TensorBoard
        if writer:
            writer.add_scalar(f'{model_name}/Loss/train_epoch', avg_train_loss, epoch + 1)
            writer.add_scalar(f'{model_name}/Loss/val_epoch', avg_val_loss, epoch + 1)
            writer.add_scalar(f'{model_name}/LR', optimizer.param_groups[0]['lr'], epoch + 1) # Log current LR

        # Learning rate scheduler step
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(model_save_dir, "best_model.pt") # Use generalized path
            torch.save(model.state_dict(), best_model_path)
            log.info(f"Epoch {epoch+1}/{epochs} complete | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Best model saved to {best_model_path}")
        else:
            log.info(f"Epoch {epoch+1}/{epochs} complete | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")


        # Check for early stopping
        if early_stopping(avg_val_loss):
            log.info(f"Early stopping triggered after {epoch+1} epochs for {model_name} model.")
            break

        # Checkpointing
        # Ensure cfg.train.checkpoint_frequency is a general config or defined per model type
        checkpoint_freq = cfg.train.get("checkpoint_frequency", 10) # Default if not in main train config
        if hasattr(train_specific_cfg, "checkpoint_frequency"): # Override with model-specific if present
             checkpoint_freq = train_specific_cfg.checkpoint_frequency

        if (epoch+1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(model_save_dir, f"checkpoint_epoch_{epoch+1}.pt") # Use generalized path
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, checkpoint_path)
            log.info(f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}")

    # Save final model
    final_model_path = os.path.join(model_save_dir, "final_model.pt") # Use generalized path
    torch.save(model.state_dict(), final_model_path)
    log.info(f"Final {model_name} model saved after {epoch+1} epochs to {final_model_path}")

    # Load best model
    best_model_path = os.path.join(model_save_dir, "best_model.pt") # Use generalized path
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        log.info(f"Loaded best {model_name} model with val loss: {best_val_loss:.6f} from {best_model_path}")
    else:
        log.warning(f"Best model path {best_model_path} not found. Current model state is from last epoch.")

    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch + 1 # This will be the actual number of epochs run
    }

    history_path = os.path.join(model_save_dir, "training_history.pt") # Use generalized path
    torch.save(history, history_path)
    log.info(f"Training history saved to {history_path}")

    return model


def validate_model_predictions( # Renamed from validate_lstm_predictions
    model: nn.Module,
    val_dataset: Dataset, # This is actually a DataLoader in the original code, should be val_loader
    cfg: DictConfig,
    device: torch.device
) -> Dict[str, Any]:
    """
    Validate model predictions and calculate metrics.

    Args:
        model: Model to evaluate (LSTM, Mamba, Transformer)
        val_loader: Validation DataLoader (Corrected type from Dataset to DataLoader)
        cfg: Configuration object
        device: Device to run on

    Returns:
        Dictionary with validation metrics
    """
    model_name = cfg.model.name.lower()
    # train_specific_cfg = cfg.train[model_name] # For batch_size for DataLoader if needed
    # model_arch_cfg = cfg.model # For other model params if needed by validation

    # Batch size for validation loader should ideally be passed in or taken from train config
    # Assuming val_dataset is actually a DataLoader as per original usage context.
    # If val_dataset is indeed a Dataset, then a DataLoader must be created here.
    # For now, let's assume val_dataset is the DataLoader (val_loader)
    if not isinstance(val_dataset, DataLoader):
        log.warning("validate_model_predictions expected a DataLoader for val_dataset, but got Dataset. Creating one.")
        # This part needs careful handling of batch_size. Where should it come from for validation?
        # Using train batch_size for now, but val batch_size could be different.
        # This should ideally be cfg.val.batch_size or cfg.train[model_name].val_batch_size
        # For now, let's assume it's available in train_specific_cfg
        val_batch_size = cfg.train[model_name].get("batch_size", 32) # Fallback, should be configured
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    else:
        val_loader = val_dataset


    model.eval()
    criterion = nn.MSELoss()

    # Collect predictions and targets
    all_inputs = []
    all_targets = []
    all_outputs = []
    val_loss = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader: # val_loader is used here
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            batch_loss = criterion(outputs, targets)
            val_loss += batch_loss.item()

            all_inputs.append(inputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    # Concatenate results
    all_inputs = np.concatenate(all_inputs, axis=0) if all_inputs else np.array([])
    all_targets = np.concatenate(all_targets, axis=0) if all_targets else np.array([])
    all_outputs = np.concatenate(all_outputs, axis=0) if all_outputs else np.array([])


    metrics = {}
    if not all_targets.size == 0 and not all_outputs.size == 0 : # Ensure there's data to calculate metrics
        # Calculate metrics
        mse = np.mean((all_targets - all_outputs) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(all_targets - all_outputs))

        # Calculate R^2 score
        # Handle cases with single sample or zero variance target for R2
        if all_targets.shape[0] > 1 and np.var(all_targets, axis=0).mean() > 1e-9: # Check for variance
            ss_tot = np.sum((all_targets - np.mean(all_targets, axis=0)) ** 2)
            ss_res = np.sum((all_targets - all_outputs) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0 # Avoid division by zero
        else:
            r2 = 0.0 # Or np.nan, depending on how you want to treat this edge case

        metrics = {
            'val_loss': val_loss / len(val_loader) if len(val_loader) > 0 else 0.0,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

        log.info(f"{model_name.upper()} Validation - Loss: {metrics.get('val_loss', 0.0):.6f}, MSE: {mse:.6f}, "
                 f"RMSE: {rmse:.6f}, MAE: {mae:.6f}, R^2: {r2:.6f}")
    else:
        log.warning(f"No data or predictions found for {model_name.upper()} validation. Skipping metrics calculation.")
        metrics['val_loss'] = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0

    return {
        'metrics': metrics,
        'inputs': all_inputs,
        'targets': all_targets,
        'predictions': all_outputs
    }
