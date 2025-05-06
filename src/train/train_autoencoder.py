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

log = logging.getLogger(__name__)

class EarlyStopping:
    """
    Early stopping handler to prevent overfitting.
    """
    def __init__(
        self, 
        patience: int = 10, 
        min_delta: float = 0.0, 
        mode: str = "min"
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in monitored value to qualify as improvement
            mode: 'min' for loss, 'max' for metrics like accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False
        
    def __call__(self, val_metric: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_metric: Current validation metric value
            
        Returns:
            True if should stop, False otherwise
        """
        if self.mode == 'min':
            score = -val_metric
        else:
            score = val_metric
            
        if self.best_score is None:
            self.best_score = score
            return False
            
        if score < self.best_score + self.min_delta:
            self.counter += 1
            log.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.counter = 0
            
        return False


def train_autoencoder(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    cfg: DictConfig,
    device: torch.device,
    writer: Optional[SummaryWriter] = None
) -> nn.Module:
    """
    Train an autoencoder model.
    
    Args:
        model: Autoencoder model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        cfg: Configuration object
        device: Device to train on
        writer: Optional TensorBoard writer
        
    Returns:
        Trained autoencoder model
    """
    # Extract training parameters
    epochs = cfg.train.autoencoder.epochs
    batch_size = cfg.model.autoencoder.batch_size
    learning_rate = cfg.model.autoencoder.learning_rate
    weight_decay = cfg.model.autoencoder.weight_decay
    
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
        for batch_idx, batch in enumerate(train_loader):
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
            
            # # Log batch progress
            # if batch_idx % cfg.train.log_interval == 0:
            #     log.info(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.6f}")
                
            #     # Log to TensorBoard
            #     if writer:
            #         global_step = epoch * len(train_loader) + batch_idx
            #         writer.add_scalar('train/batch_loss', loss.item(), global_step)
        
        # Calculate average epoch loss
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for val_batch in val_loader:
                val_inputs = val_batch[0].to(device)
                val_outputs = model(val_inputs)
                val_batch_loss = criterion(val_outputs, val_inputs)
                val_loss += val_batch_loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Log epoch results
        log.info(f"Epoch {epoch+1}/{epochs} complete | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # Log to TensorBoard
        if writer:
            writer.add_scalar('train/epoch_loss', avg_train_loss, epoch)
            writer.add_scalar('val/loss', avg_val_loss, epoch)
            
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
            log.info(f"New best model saved with val loss: {best_val_loss:.6f}")
        
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
