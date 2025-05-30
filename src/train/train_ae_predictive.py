#!/usr/bin/env python
# src/train/train_ae_lstm.py - AE-LSTM combined model training module

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
from src.model.ae_predictive import AEPredictiveModel # For type hinting
from src.train.train_autoencoder import EarlyStopping  # Reuse the EarlyStopping class

log = logging.getLogger(__name__)

def train_ae_predictive( # Renamed from train_ae_lstm
    model: AEPredictiveModel, # Changed type hint
    train_dataset: Dataset,
    val_dataset: Dataset,
    cfg: DictConfig,
    device: torch.device,
    writer: Optional[SummaryWriter] = None
) -> AEPredictiveModel: # Changed return type hint
    """
    Train a combined Autoencoder-Predictive model.

    Args:
        model: AE-Predictive model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        cfg: Configuration object
        device: Device to train on
        writer: Optional TensorBoard writer

    Returns:
        Trained AE-Predictive model
    """
    # Extract training parameters - uses ae_predictive section now
    train_cfg = cfg.train.ae_predictive
    epochs = train_cfg.epochs
    batch_size = train_cfg.batch_size
    learning_rate = train_cfg.learning_rate
    weight_decay = train_cfg.weight_decay

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.data.loader.num_workers,
        pin_memory=cfg.data.loader.pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.data.loader.num_workers,
        pin_memory=cfg.data.loader.pin_memory
    )

    # Create optimizer with appropriate parameters
    # Only optimize unfrozen parameters if autoencoder is frozen
    params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(
        params_to_optimize,
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Create learning rate scheduler
    scheduler_cfg = train_cfg.scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=scheduler_cfg.factor, # Use .get for robustness or ensure defined
        patience=scheduler_cfg.patience,
        min_lr=scheduler_cfg.min_lr
        # verbose=True # Removed, not a valid argument for ReduceLROnPlateau in current PyTorch
    )

    # Loss function (MSE for prediction)
    criterion = nn.MSELoss()

    # Early stopping
    early_stopping_cfg = train_cfg.early_stopping
    early_stopping = EarlyStopping(
        patience=early_stopping_cfg.patience,
        min_delta=early_stopping_cfg.min_delta,
        mode='min'
    )

    # Create directories for saving models (path remains cfg.paths.combined_model_dir for now)
    # This path could be made dynamic based on model.name if multiple combined model types were trained.
    model_save_dir = cfg.paths.combined_model_dir
    os.makedirs(model_save_dir, exist_ok=True)

    # Training loop
    active_model_name = cfg.model.name # e.g. lstm, mamba, transformer
    log.info(f"Beginning AE-{active_model_name.upper()} combined training for {epochs} epochs. Saving to {model_save_dir}")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        # Training
        for batch_idx, (inputs, targets) in enumerate(train_loader):
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

            # Log batch progress
            # Ensure cfg.train.log_interval exists or use a default.
            log_interval = cfg.train.get("log_interval", 10)
            if batch_idx % log_interval == 0:
                log.info(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.6f}")

                # Log to TensorBoard
                if writer:
                    global_step = epoch * len(train_loader) + batch_idx
                    writer.add_scalar(f'AE-{active_model_name}/train/batch_loss', loss.item(), global_step)

        # Calculate average epoch loss
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs = model(val_inputs)
                val_batch_loss = criterion(val_outputs, val_targets)
                val_loss += val_batch_loss.item()

                # Collect predictions and targets for metrics
                all_targets.append(val_targets.cpu().numpy())
                all_predictions.append(val_outputs.cpu().numpy())

        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Log epoch results
        log.info(f"Epoch {epoch+1}/{epochs} complete | AE-{active_model_name.upper()} Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # Log to TensorBoard
        if writer:
            writer.add_scalar(f'AE-{active_model_name}/train/epoch_loss', avg_train_loss, epoch)
            writer.add_scalar(f'AE-{active_model_name}/val/loss', avg_val_loss, epoch)

            # Calculate and log additional metrics
            # Ensure cfg.train.log_metrics_epoch_interval or similar exists if used
            log_metrics_epoch_interval = cfg.train.get("log_metrics_epoch_interval", 5)
            if epoch % log_metrics_epoch_interval == 0 and all_targets and all_predictions:
                # Combine batch predictions and targets
                all_targets_np = np.concatenate(all_targets, axis=0) # Renamed to avoid conflict
                all_predictions_np = np.concatenate(all_predictions, axis=0) # Renamed

                # Calculate MSE and MAE
                mse = np.mean((all_targets_np - all_predictions_np) ** 2)
                mae = np.mean(np.abs(all_targets_np - all_predictions_np))

                writer.add_scalar(f'AE-{active_model_name}/val/mse', mse, epoch)
                writer.add_scalar(f'AE-{active_model_name}/val/mae', mae, epoch)

                # Log latent space visualizations if possible (model here is AEPredictiveModel)
                # The AEPredictiveModel itself doesn't have an encode method directly.
                # It would be model.autoencoder.encode(...)
                log_latent_epoch_interval = cfg.train.get("log_latent_epoch_interval", 10)
                if hasattr(model, 'autoencoder') and hasattr(model.autoencoder, 'encode') and epoch % log_latent_epoch_interval == 0:
                    try:
                        # Assuming val_loader yields (inputs, targets) where inputs are for AE
                        sample_ae_inputs = next(iter(val_loader))[0][:10].to(device) # Taking raw inputs for AE

                        # Reshape if necessary for AE's encode method
                        # This part needs to be careful about the shape expected by autoencoder.encode
                        # If val_loader provides sequences, and AE expects single items:
                        if sample_ae_inputs.ndim == 3 and model.autoencoder.input_dim == sample_ae_inputs.shape[2] : # (B, S, F)
                             sample_ae_inputs_flat = sample_ae_inputs.reshape(-1, model.autoencoder.input_dim)
                             latent_codes = model.autoencoder.encode(sample_ae_inputs_flat)
                        else: # Assuming AE takes (B, F)
                             latent_codes = model.autoencoder.encode(sample_ae_inputs)

                        if latent_codes.dim() == 2 and latent_codes.size(1) >= 2: # Check if latent codes are suitable for embedding projector
                            writer.add_embedding(
                                latent_codes,
                                # metadata=list(range(latent_codes.size(0))), # Metadata might need adjustment
                                tag=f'AE-{active_model_name}/latent_space',
                                global_step=epoch
                            )
                    except Exception as e:
                        log.warning(f"Could not log latent space visualization for AE-{active_model_name}: {e}")

        # Learning rate scheduler step
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(model_save_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            log.info(f"New best AE-{active_model_name.upper()} model saved with val loss: {best_val_loss:.6f} to {best_model_path}")

        # Check for early stopping
        if early_stopping(avg_val_loss):
            log.info(f"Early stopping triggered for AE-{active_model_name.upper()} after {epoch+1} epochs")
            break

        # Checkpointing
        checkpoint_freq = cfg.train.get("checkpoint_frequency", 10) # General checkpoint freq
        if hasattr(train_cfg, "checkpoint_frequency"): # Override with model-specific if present
             checkpoint_freq = train_cfg.checkpoint_frequency

        if (epoch+1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(model_save_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, checkpoint_path)
            log.info(f"AE-{active_model_name.upper()} checkpoint saved at epoch {epoch+1} to {checkpoint_path}")

    # Save final model
    final_model_path = os.path.join(model_save_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    log.info(f"Final AE-{active_model_name.upper()} model saved after {epoch+1} epochs to {final_model_path}")

    # Load best model
    best_model_path = os.path.join(model_save_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        log.info(f"Loaded best AE-{active_model_name.upper()} model with val loss: {best_val_loss:.6f} from {best_model_path}")
    else:
        log.warning(f"Best model path {best_model_path} not found for AE-{active_model_name.upper()}. Current model state is from last epoch.")

    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch + 1 # Actual number of epochs run
    }

    history_path = os.path.join(model_save_dir, "training_history.pt")
    torch.save(history, history_path)
    log.info(f"AE-{active_model_name.upper()} training history saved to {history_path}")

    return model
