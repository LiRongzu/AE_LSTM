#!/usr/bin/env python
# src/visualization/visualize.py - Visualization utilities for model outputs

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from typing import Dict, List, Any, Optional, Union, Tuple
from omegaconf import DictConfig
import logging
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

log = logging.getLogger(__name__)

def plot_reconstruction_samples(
    autoencoder: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    cfg: DictConfig,
    device: torch.device,
    n_samples: int = 5
) -> None:
    """
    Plot samples of original vs reconstructed data from the autoencoder.
    
    Args:
        autoencoder: Trained autoencoder model
        dataset: Dataset containing samples
        cfg: Configuration object
        device: Device to run inference on
        n_samples: Number of samples to plot
    """
    # Create output directory
    output_dir = os.path.join(cfg.paths.visualization_dir, "reconstructions")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    autoencoder.eval()
    
    # Select random samples
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get original data
            original = dataset[idx]
            if isinstance(original, tuple):
                original = original[0]  # Take input part if dataset returns (input, target)
                
            # Add batch dimension if needed
            if len(original.shape) == 2:
                original = original.unsqueeze(0)
                
            # Reconstruct data
            original = original.to(device)
            reconstructed = autoencoder(original)
            
            # Move to CPU and convert to numpy for plotting
            original = original.cpu().numpy().squeeze()
            reconstructed = reconstructed.cpu().numpy().squeeze()
            
            # Calculate error
            error = np.abs(reconstructed - original)
            
            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Determine color scale based on data
            vmin = min(original.min(), reconstructed.min())
            vmax = max(original.max(), reconstructed.max())
            
            # Plot original
            im0 = axes[0].imshow(original, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[0].set_title("Original")
            axes[0].axis('off')
            divider = make_axes_locatable(axes[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im0, cax=cax)
            
            # Plot reconstruction
            im1 = axes[1].imshow(reconstructed, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[1].set_title("Reconstructed")
            axes[1].axis('off')
            divider = make_axes_locatable(axes[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im1, cax=cax)
            
            # Plot error
            im2 = axes[2].imshow(error, cmap='hot')
            axes[2].set_title(f"Error (MAE={error.mean():.4f})")
            axes[2].axis('off')
            divider = make_axes_locatable(axes[2])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im2, cax=cax)
            
            # Set figure title
            fig.suptitle(f"Sample {i+1}")
            fig.tight_layout()
            
            # Save figure
            fig.savefig(os.path.join(output_dir, f"reconstruction_sample_{i+1}.png"), dpi=300)
            plt.close(fig)
    
    log.info(f"Saved {n_samples} reconstruction samples to {output_dir}")

def plot_prediction_samples(
    ae_lstm_model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    cfg: DictConfig,
    device: torch.device,
    n_samples: int = 5
) -> None:
    """
    Plot samples of ground truth vs predicted data from the AE-LSTM model.
    
    Args:
        ae_lstm_model: Trained AE-LSTM model
        dataset: Dataset containing samples
        cfg: Configuration object
        device: Device to run inference on
        n_samples: Number of samples to plot
    """
    # Create output directory
    output_dir = os.path.join(cfg.paths.visualization_dir, "predictions")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    ae_lstm_model.eval()
    
    # Select random samples - making sure to get sequences
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get sample
            if hasattr(dataset, "__getitem__"):
                x_seq, y_true = dataset[idx]
                if not isinstance(x_seq, torch.Tensor):
                    x_seq = torch.tensor(x_seq)
                if not isinstance(y_true, torch.Tensor):
                    y_true = torch.tensor(y_true)
                
                # Add batch dimension if needed
                if len(x_seq.shape) == 2:
                    x_seq = x_seq.unsqueeze(0)
                
                # Get prediction
                x_seq = x_seq.to(device)
                y_pred = ae_lstm_model(x_seq)
                
                # Move to CPU and convert to numpy for plotting
                y_true = y_true.cpu().numpy().squeeze()
                y_pred = y_pred.cpu().numpy().squeeze()
            else:
                log.warning("Dataset does not support indexing, skipping prediction plot")
                return
            
            # Calculate error
            error = np.abs(y_pred - y_true)
            
            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Determine color scale based on data
            vmin = min(y_true.min(), y_pred.min())
            vmax = max(y_true.max(), y_pred.max())
            
            # Plot ground truth
            im0 = axes[0].imshow(y_true, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[0].set_title("Ground Truth")
            axes[0].axis('off')
            divider = make_axes_locatable(axes[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im0, cax=cax)
            
            # Plot prediction
            im1 = axes[1].imshow(y_pred, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[1].set_title("Predicted")
            axes[1].axis('off')
            divider = make_axes_locatable(axes[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im1, cax=cax)
            
            # Plot error
            im2 = axes[2].imshow(error, cmap='hot')
            axes[2].set_title(f"Error (MAE={error.mean():.4f})")
            axes[2].axis('off')
            divider = make_axes_locatable(axes[2])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im2, cax=cax)
            
            # Set figure title
            fig.suptitle(f"Sample {i+1}")
            fig.tight_layout()
            
            # Save figure
            fig.savefig(os.path.join(output_dir, f"prediction_sample_{i+1}.png"), dpi=300)
            plt.close(fig)
    
    log.info(f"Saved {n_samples} prediction samples to {output_dir}")

def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    cfg: DictConfig,
    model_type: str = "autoencoder"
) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        cfg: Configuration object
        model_type: Type of model ("autoencoder", "lstm", or "ae_lstm")
    """
    # Create output directory
    output_dir = os.path.join(cfg.paths.visualization_dir, "loss_curves")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Training Loss')
    ax.plot(epochs, val_losses, label='Validation Loss')
    
    # Add labels and title
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title(f'{model_type.upper()} Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save figure
    fig.savefig(os.path.join(output_dir, f"{model_type}_loss.png"), dpi=300)
    plt.close(fig)
    
    log.info(f"Saved {model_type} loss curves to {output_dir}")

def plot_metrics(
    ae_metrics: Dict[str, float],
    lstm_metrics: Dict[str, float],
    cfg: DictConfig
) -> None:
    """
    Plot evaluation metrics for both autoencoder and LSTM models.
    
    Args:
        ae_metrics: Dictionary of autoencoder evaluation metrics
        lstm_metrics: Dictionary of LSTM evaluation metrics
        cfg: Configuration object
    """
    # Create output directory
    output_dir = os.path.join(cfg.paths.visualization_dir, "metrics")
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine metrics
    metrics = {
        'Autoencoder': ae_metrics,
        'AE-LSTM': lstm_metrics
    }
    
    # Get common metrics
    all_metrics = set()
    for model_metrics in metrics.values():
        all_metrics.update(model_metrics.keys())
    
    # Create bar plot for each metric
    for metric in all_metrics:
        # Skip metrics that aren't present in both models
        if not all(metric in model_metrics for model_metrics in metrics.values()):
            continue
            
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Get values
        values = [metrics[model][metric] for model in metrics]
        
        # Plot bars
        bars = ax.bar(list(metrics.keys()), values)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # Add labels and title
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} by Model')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Save figure
        fig.savefig(os.path.join(output_dir, f"{metric}_comparison.png"), dpi=300)
        plt.close(fig)
    
    # Create table with all metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.axis('tight')
    
    # Create table data
    table_data = []
    for metric in sorted(all_metrics):
        row = [metric]
        for model in metrics.keys():
            if metric in metrics[model]:
                row.append(f"{metrics[model][metric]:.4f}")
            else:
                row.append("N/A")
        table_data.append(row)
    
    # Create the table
    table = ax.table(
        cellText=table_data,
        colLabels=["Metric"] + list(metrics.keys()),
        loc='center',
        cellLoc='center'
    )
    
    # Adjust table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Save figure
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "metrics_table.png"), dpi=300)
    plt.close(fig)
    
    log.info(f"Saved metrics visualization to {output_dir}")
