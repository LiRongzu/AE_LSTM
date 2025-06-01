#!/usr/bin/env python
# src/train/train_autoencoder.py - Autoencoder training module

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
from tqdm import tqdm

log = logging.getLogger(__name__)

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.early_stop = False
        if self.mode == "min":
            self.best_metric_val = float('inf') # 直接存储最佳的度量值
        else: # mode == "max"
            self.best_metric_val = float('-inf')

    def __call__(self, current_metric_val: float) -> bool:
        has_improved = False
        if self.mode == "min":
            # 如果当前度量值比历史最佳值改善了（考虑 min_delta）
            if current_metric_val < self.best_metric_val - self.min_delta:
                self.best_metric_val = current_metric_val
                has_improved = True
        else: # mode == "max"
            if current_metric_val > self.best_metric_val + self.min_delta:
                self.best_metric_val = current_metric_val
                has_improved = True

        if has_improved:
            self.counter = 0
        else:
            self.counter += 1
            # log.info(f"EarlyStopping counter: {self.counter} out of {self.patience}") # 只在未改善时打印

        if self.counter >= self.patience:
            self.early_stop = True
            return True # 应该停止

        return False # 不应该停止


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: DictConfig,
    device: torch.device,
    writer: Optional[SummaryWriter] = None
) -> nn.Module:
    """
    Train an autoencoder model.
    
    Args:
        model: Autoencoder model to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        cfg: Configuration object
        device: Device to train on
        writer: Optional TensorBoard writer
        
    Returns:
        Trained autoencoder model
    """
    # Extract training parameters
    epochs = cfg.train.autoencoder.epochs
    learning_rate = cfg.train.autoencoder.learning_rate
    weight_decay = cfg.train.autoencoder.weight_decay

    
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
        factor=cfg.train.autoencoder.scheduler.factor,
        patience=cfg.train.autoencoder.scheduler.patience,
        min_lr=cfg.train.autoencoder.scheduler.min_lr,
        verbose=True
    )
    
    # Loss function (MSE for reconstruction)
    criterion = nn.MSELoss()
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=cfg.train.autoencoder.early_stopping.patience,
        min_delta=cfg.train.autoencoder.early_stopping.min_delta,
        mode='min'
    )
    
    # Create directories for saving models
    os.makedirs(cfg.paths.ae_model_dir, exist_ok=True)
    
    # Training loop
    log.info(f"Beginning autoencoder training for {epochs} epochs")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        # Training
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False, unit="batch")
        for batch_idx, batch in enumerate(train_progress_bar):
            # Get data
            inputs = batch[0].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            
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
                writer.add_scalar(f'{cfg.model.autoencoder.type}/Loss/train_batch', loss.item(), current_iter)
        
        # Calculate average epoch loss
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False, unit="batch")
        with torch.no_grad():
            for val_batch in val_progress_bar:
                val_inputs = val_batch[0].to(device)
                val_outputs = model(val_inputs)
                val_batch_loss = criterion(val_outputs, val_inputs)
                val_loss += val_batch_loss.item()
                val_progress_bar.set_postfix(loss=f"{val_batch_loss.item():.6f}")
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Log epoch results
        # log.info(f"Epoch {epoch+1}/{epochs} complete | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # Log to TensorBoard
        if writer:
            writer.add_scalar(f'{cfg.model.autoencoder.type}/Loss/train_epoch', avg_train_loss, epoch + 1)
            writer.add_scalar(f'{cfg.model.autoencoder.type}/Loss/val_epoch', avg_val_loss, epoch + 1)
            if scheduler: # If you have a learning rate scheduler
                writer.add_scalar(f'{cfg.model.autoencoder.type}/LR', scheduler.get_last_lr()[0], epoch + 1)
            else:
                writer.add_scalar(f'{cfg.model.autoencoder.type}/LR', optimizer.param_groups[0]['lr'], epoch + 1)
            
            # Log sample reconstructions
            if epoch % 5 == 0:
                with torch.no_grad():
                    # Get a few samples
                    samples = next(iter(val_loader))[0][:4].to(device)
                    reconstructed = model(samples)
                    
                    # Convert to numpy for visualization
                    samples_np = samples.cpu().numpy()
                    reconstructed_np = reconstructed.cpu().numpy()
                    
                    # Log to TensorBoard as images if they have spatial structure
                    # This assumes the data can be reshaped to a grid
                    if cfg.visualization.get('reshape_to_grid', False):
                        height, width = cfg.visualization.get('grid_height', 50), cfg.visualization.get('grid_width', 40)
                        for i in range(min(4, samples.size(0))):
                            sample_img = samples_np[i].reshape(height, width)
                            recon_img = reconstructed_np[i].reshape(height, width)
                            error_img = np.abs(sample_img - recon_img)
                            
                            writer.add_image(f'sample_{i}/original', sample_img, epoch, dataformats='HW')
                            writer.add_image(f'sample_{i}/reconstructed', recon_img, epoch, dataformats='HW')
                            writer.add_image(f'sample_{i}/error', error_img, epoch, dataformats='HW')
        
        # Learning rate scheduler step
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(cfg.paths.ae_model_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            log.info(f"Epoch {epoch+1}/{epochs} complete | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            # log.info(f"New best model saved with val loss: {best_val_loss:.6f}")
        
        # Check for early stopping
        if early_stopping(avg_val_loss):
            log.info(f"Early stopping triggered after {epoch+1} epochs")
            break
            
        # Checkpointing
        if (epoch+1) % cfg.train.checkpoint_frequency == 0:
            checkpoint_path = os.path.join(cfg.paths.ae_model_dir, f"checkpoint_epoch_{epoch+1}.pt")
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
    final_model_path = os.path.join(cfg.paths.ae_model_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    log.info(f"Final model saved after {epoch+1} epochs")
    
    # Load best model
    best_model_path = os.path.join(cfg.paths.ae_model_dir, "best_model.pt")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    log.info(f"Loaded best model with val loss: {best_val_loss:.6f}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch + 1
    }
    
    history_path = os.path.join(cfg.paths.ae_model_dir, "training_history.pt")
    torch.save(history, history_path)
    
    return model
