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
from src.train.train_autoencoder import EarlyStopping  # Reuse the EarlyStopping class

log = logging.getLogger(__name__)

def train_ae_lstm(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    cfg: DictConfig,
    device: torch.device,
    writer: Optional[SummaryWriter] = None
) -> nn.Module:
    """
    Train a combined AE-LSTM model.
    
    Args:
        model: AE-LSTM model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        cfg: Configuration object
        device: Device to train on
        writer: Optional TensorBoard writer
        
    Returns:
        Trained AE-LSTM model
    """
    # Extract training parameters
    epochs = cfg.train.ae_lstm.epochs
    batch_size = cfg.train.ae_lstm.batch_size
    learning_rate = cfg.train.ae_lstm.learning_rate
    weight_decay = cfg.train.ae_lstm.weight_decay
    
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=cfg.train.ae_lstm.scheduler.factor,
        patience=cfg.train.ae_lstm.scheduler.patience,
        min_lr=cfg.train.ae_lstm.scheduler.min_lr,
        verbose=True
    )
    
    # Loss function (MSE for prediction)
    criterion = nn.MSELoss()
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=cfg.train.ae_lstm.early_stopping.patience,
        min_delta=cfg.train.ae_lstm.early_stopping.min_delta,
        mode='min'
    )
    
    # Create directories for saving models
    os.makedirs(cfg.paths.combined_model_dir, exist_ok=True)
    
    # Training loop
    log.info(f"Beginning AE-LSTM training for {epochs} epochs")
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
            if batch_idx % cfg.train.log_interval == 0:
                log.info(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.6f}")
                
                # Log to TensorBoard
                if writer:
                    global_step = epoch * len(train_loader) + batch_idx
                    writer.add_scalar('train/batch_loss', loss.item(), global_step)
        
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
        log.info(f"Epoch {epoch+1}/{epochs} complete | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # Log to TensorBoard
        if writer:
            writer.add_scalar('train/epoch_loss', avg_train_loss, epoch)
            writer.add_scalar('val/loss', avg_val_loss, epoch)
            
            # Calculate and log additional metrics
            if epoch % 5 == 0 and all_targets and all_predictions:
                # Combine batch predictions and targets
                all_targets = np.concatenate(all_targets, axis=0)
                all_predictions = np.concatenate(all_predictions, axis=0)
                
                # Calculate MSE and MAE
                mse = np.mean((all_targets - all_predictions) ** 2)
                mae = np.mean(np.abs(all_targets - all_predictions))
                
                writer.add_scalar('val/mse', mse, epoch)
                writer.add_scalar('val/mae', mae, epoch)
                
                # Log latent space visualizations if possible
                if hasattr(model, 'encode') and epoch % 10 == 0:
                    try:
                        sample_inputs = next(iter(val_loader))[0][:10].to(device)
                        latent_codes = model.encode(sample_inputs)
                        
                        if latent_codes.dim() == 2 and latent_codes.size(1) == 2:
                            # Visualize 2D latent space directly
                            writer.add_embedding(
                                latent_codes,
                                metadata=list(range(latent_codes.size(0))),
                                global_step=epoch
                            )
                    except Exception as e:
                        log.warning(f"Could not log latent space visualization: {e}")
        
        # Learning rate scheduler step
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(cfg.paths.combined_model_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            log.info(f"New best model saved with val loss: {best_val_loss:.6f}")
        
        # Check for early stopping
        if early_stopping(avg_val_loss):
            log.info(f"Early stopping triggered after {epoch+1} epochs")
            break
            
        # Checkpointing
        if (epoch+1) % cfg.train.checkpoint_frequency == 0:
            checkpoint_path = os.path.join(cfg.paths.combined_model_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, checkpoint_path)
            log.info(f"Checkpoint saved at epoch {epoch+1}")
    
    # Save final model
    final_model_path = os.path.join(cfg.paths.combined_model_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    log.info(f"Final model saved after {epoch+1} epochs")
    
    # Load best model
    best_model_path = os.path.join(cfg.paths.combined_model_dir, "best_model.pt")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    log.info(f"Loaded best model with val loss: {best_val_loss:.6f}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch + 1
    }
    
    history_path = os.path.join(cfg.paths.combined_model_dir, "training_history.pt")
    torch.save(history, history_path)
    
    return model
