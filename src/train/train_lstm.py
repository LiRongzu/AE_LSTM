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
from src.train.train_autoencoder import EarlyStopping  # Reuse the EarlyStopping class

log = logging.getLogger(__name__)

def train_lstm(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    cfg: DictConfig,
    device: torch.device,
    writer: Optional[SummaryWriter] = None
) -> nn.Module:
    """
    Train an LSTM model.
    
    Args:
        model: LSTM model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        cfg: Configuration object
        device: Device to train on
        writer: Optional TensorBoard writer
        
    Returns:
        Trained LSTM model
    """
    # Extract training parameters
    epochs = cfg.train.lstm.epochs
    batch_size = cfg.model.lstm.batch_size
    learning_rate = cfg.model.lstm.learning_rate
    weight_decay = cfg.model.lstm.weight_decay
    
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
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=cfg.train.lstm.scheduler.factor,
        patience=cfg.train.lstm.scheduler.patience,
        min_lr=cfg.train.lstm.scheduler.min_lr,
        verbose=True
    )
    
    # Loss function (MSE for prediction)
    criterion = nn.MSELoss()
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=cfg.train.lstm.early_stopping.patience,
        min_delta=cfg.train.lstm.early_stopping.min_delta,
        mode='min'
    )
    
    # Create directories for saving models
    os.makedirs(cfg.paths.lstm_model_dir, exist_ok=True)
    
    # Training loop
    log.info(f"Beginning LSTM training for {epochs} epochs")
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
        
        # Learning rate scheduler step
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(cfg.paths.lstm_model_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            log.info(f"New best model saved with val loss: {best_val_loss:.6f}")
        
        # Check for early stopping
        if early_stopping(avg_val_loss):
            log.info(f"Early stopping triggered after {epoch+1} epochs")
            break
            
        # Checkpointing
        if (epoch+1) % cfg.train.checkpoint_frequency == 0:
            checkpoint_path = os.path.join(cfg.paths.lstm_model_dir, f"checkpoint_epoch_{epoch+1}.pt")
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
    final_model_path = os.path.join(cfg.paths.lstm_model_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    log.info(f"Final model saved after {epoch+1} epochs")
    
    # Load best model
    best_model_path = os.path.join(cfg.paths.lstm_model_dir, "best_model.pt")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    log.info(f"Loaded best model with val loss: {best_val_loss:.6f}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch + 1
    }
    
    history_path = os.path.join(cfg.paths.lstm_model_dir, "training_history.pt")
    torch.save(history, history_path)
    
    return model


def validate_lstm_predictions(
    model: nn.Module,
    val_dataset: Dataset,
    cfg: DictConfig,
    device: torch.device
) -> Dict[str, Any]:
    """
    Validate LSTM predictions and calculate metrics.
    
    Args:
        model: LSTM model to evaluate
        val_dataset: Validation dataset
        cfg: Configuration object
        device: Device to run on
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    criterion = nn.MSELoss()
    
    # Create data loader
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.model.lstm.batch_size, 
        shuffle=False
    )
    
    # Collect predictions and targets
    all_inputs = []
    all_targets = []
    all_outputs = []
    val_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            batch_loss = criterion(outputs, targets)
            val_loss += batch_loss.item()
            
            all_inputs.append(inputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
    
    # Concatenate results
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)
    
    # Calculate metrics
    mse = np.mean((all_targets - all_outputs) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_targets - all_outputs))
    
    # Calculate R^2 score
    ss_tot = np.sum((all_targets - np.mean(all_targets, axis=0)) ** 2)
    ss_res = np.sum((all_targets - all_outputs) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    metrics = {
        'val_loss': val_loss / len(val_loader),
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    log.info(f"LSTM Validation - Loss: {metrics['val_loss']:.6f}, MSE: {mse:.6f}, "
             f"RMSE: {rmse:.6f}, MAE: {mae:.6f}, R^2: {r2:.6f}")
    
    return {
        'metrics': metrics,
        'inputs': all_inputs,
        'targets': all_targets,
        'predictions': all_outputs
    }
