#!/usr/bin/env python
# src/visualization/visualize.py - Visualization utilities for model outputs

import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from omegaconf import DictConfig

log = logging.getLogger(__name__)

def load_triangulation_data(cfg: DictConfig) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Loads vertices and triangles for tripcolor plotting."""
    vertices_path = None
    triangles_path = None

    # Construct full paths using cfg.paths.data_dir and cfg.data specific file names
    base_data_dir = cfg.paths.get('data_dir', '.') # Default to current dir if not set

    vertices_file = cfg.data.get('vertices_file', 'vertices.npy')
    triangles_file = cfg.data.get('triangles_file', 'triangles.npy')

    if os.path.isabs(vertices_file):
        vertices_path = vertices_file
    else:
        vertices_path = os.path.join(base_data_dir, vertices_file)

    if os.path.isabs(triangles_file):
        triangles_path = triangles_file
    else:
        triangles_path = os.path.join(base_data_dir, triangles_file)

    vertices, triangles = None, None
    try:
        if os.path.exists(vertices_path):
            vertices = np.load(vertices_path)
            log.info(f"Loaded vertices from {vertices_path}, shape: {vertices.shape}")
        else:
            log.error(f"Vertices file not found at {vertices_path}")
            # vertices will remain None if not assigned

        if os.path.exists(triangles_path):
            triangles = np.load(triangles_path)
            log.info(f"Loaded triangles from {triangles_path}, shape: {triangles.shape}")
        else:
            log.error(f"Triangles file not found at {triangles_path}")
            # triangles will remain None if not assigned
            
    except Exception as e:
        log.error(f"Error loading triangulation data: {e}")
        vertices, triangles = None, None # Ensure they are None if loading fails
    
    # Process vertices if loaded successfully
    if vertices is not None:
        if vertices.ndim == 2 and vertices.shape[0] == 2 and vertices.shape[1] > 0: # Check for (2, N) shape
            log.info(f"Transposing vertices from {vertices.shape} to ({vertices.shape[1]}, {vertices.shape[0]}) for plotting.")
            vertices = vertices.T
        # After potential transpose, vertices should be (N, 2)
        if not (vertices.ndim == 2 and vertices.shape[1] == 2):
            log.error(f"Vertices are not in the expected (N, 2) shape after processing. Got shape: {vertices.shape}.")
            vertices = None # Invalidate if shape is wrong
        else:
            log.info(f"Vertices shape after processing: {vertices.shape}")

    # Process triangles if loaded successfully
    if triangles is not None:
        try:
            if not np.issubdtype(triangles.dtype, np.number):
                log.error(f"Triangles data type is not numeric ({triangles.dtype}), cannot subtract 1.")
                triangles = None # Invalidate
            elif triangles.size == 0:
                log.warning("Triangles array is empty.")
                # Depending on requirements, you might want to invalidate or allow empty arrays.
                # For now, let it pass but it might cause issues downstream if not handled.
            else:
                # Subtract 1 for 0-based indexing and cast to integer type
                triangles = (triangles - 1).astype(np.int_)
                log.info(f"Triangles after subtracting 1 and casting to int: shape {triangles.shape}, dtype {triangles.dtype}")
        except Exception as e:
            log.error(f"Error processing triangles (subtracting 1 and casting to int): {e}")
            triangles = None # Invalidate on error
        
        if triangles is not None: # Re-check after try-except
            if not (triangles.ndim == 2 and triangles.shape[1] == 3):
                log.error(f"Triangles are not in the expected (M, 3) shape after processing. Got shape: {triangles.shape}.")
                triangles = None # Invalidate
            elif not np.issubdtype(triangles.dtype, np.integer):
                 log.error(f"Triangles dtype is not integer after processing. Got dtype: {triangles.dtype}.")
                 triangles = None # Invalidate

    # Final check before returning
    if vertices is None:
        log.warning("Returning None for vertices.")
    if triangles is None:
        log.warning("Returning None for triangles.")
        
    return vertices, triangles

def plot_reconstruction_samples(
    model: torch.nn.Module,
    test_loader,
    cfg: DictConfig,
    device: torch.device,
    n_samples: int = 5,
    save_dir: Optional[str] = None
):
    """
    Plots original vs. reconstructed samples using tripcolor for triangular meshes.
    """
    model.eval()
    vertices, triangles = load_triangulation_data(cfg)

    if vertices is None or triangles is None:
        log.error("Cannot plot reconstruction samples: Missing vertices or triangles data.")
        return

    samples_collected = 0
    # Adjust subplot layout if needed, tripcolor might need more space
    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 3 * n_samples)) 
    if n_samples == 1:
        axes = np.array([axes])

    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            if samples_collected >= n_samples:
                break
            
            if isinstance(batch_data, list) or isinstance(batch_data, tuple):
                inputs = batch_data[0].to(device)
            else:
                inputs = batch_data.to(device)

            if inputs.ndim == 1:
                inputs = inputs.unsqueeze(0)
            
            reconstructions = model(inputs)

            for j in range(inputs.size(0)):
                if samples_collected >= n_samples:
                    break

                original_values = inputs[j].cpu().numpy() # Shape (n_features,) or (1, n_features)
                reconstructed_values = reconstructions[j].cpu().numpy() # Shape (n_features,) or (1, n_features)

                # Ensure values are 1D array of size num_vertices
                if original_values.ndim > 1: original_values = original_values.squeeze()
                if reconstructed_values.ndim > 1: reconstructed_values = reconstructed_values.squeeze()

                if original_values.shape[0] != vertices.shape[0]:
                    log.error(f"Sample {samples_collected+1}: Number of features ({original_values.shape[0]}) "
                              f"does not match number of vertices ({vertices.shape[0]}). Cannot use tripcolor.")
                    samples_collected += 1 # Increment to avoid infinite loop if all samples are bad
                    continue
                
                ax_orig = axes[samples_collected, 0]
                ax_recon = axes[samples_collected, 1]

                vmin = min(original_values.min(), reconstructed_values.min())
                vmax = max(original_values.max(), reconstructed_values.max())

                # Plot original
                tpc_orig = ax_orig.tripcolor(vertices[:, 0], vertices[:, 1], triangles, original_values,
                                             cmap='viridis', vmin=vmin, vmax=vmax, shading='gouraud')
                ax_orig.set_title(f"Original Sample {samples_collected+1}")
                ax_orig.set_xlabel("Longitude")
                ax_orig.set_ylabel("Latitude")
                ax_orig.axis('equal') # Keep aspect ratio for geo data

                # Plot reconstructed
                tpc_recon = ax_recon.tripcolor(vertices[:, 0], vertices[:, 1], triangles, reconstructed_values,
                                               cmap='viridis', vmin=vmin, vmax=vmax, shading='gouraud')
                ax_recon.set_title(f"Reconstructed Sample {samples_collected+1}")
                ax_recon.set_xlabel("Longitude")
                ax_recon.set_ylabel("Latitude")
                ax_recon.axis('equal')
                
                samples_collected += 1
    
    if samples_collected == 0:
        log.warning("No samples were collected/plotted for reconstruction. Skipping plot saving.")
        plt.close(fig)
        return

    # Add a single colorbar for the figure, linked to the last reconstruction plot
    # Or create separate colorbars if preferred
    fig.colorbar(tpc_recon, ax=axes[:, 1], orientation='vertical', fraction=0.046, pad=0.04)
    plt.tight_layout(rect=[0, 0, 0.95, 1]) # Adjust layout to make space for colorbar

    if save_dir is None:
        save_dir = os.path.join(cfg.paths.visualization_dir, "reconstructions")
    os.makedirs(save_dir, exist_ok=True)
    
    plot_filename = os.path.join(save_dir, f"{cfg.experiment.name}_reconstruction_tripcolor.png")
    plt.savefig(plot_filename)
    log.info(f"Reconstruction (tripcolor) samples plot saved to {plot_filename}")
    plt.close(fig)

def plot_prediction_samples(
    model: torch.nn.Module,
    test_loader: DataLoader, # Changed from dataset to loader
    cfg: DictConfig,
    device: torch.device,
    n_samples: int = 5,
    save_dir: Optional[str] = None
):
    """
    Plots true future values vs. predicted future values using tripcolor.
    Assumes model output and targets are feature vectors corresponding to vertices.
    """
    model.eval()
    vertices, triangles = load_triangulation_data(cfg)

    if vertices is None or triangles is None:
        log.error("Cannot plot prediction samples: Missing vertices or triangles data.")
        return

    samples_collected = 0
    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 3 * n_samples))
    if n_samples == 1:
        axes = np.array([axes])
        
    # Get the scaler for inverse transform if it exists and is needed
    # This depends on whether your targets/predictions are scaled or in original units
    # For simplicity, assuming they are in units suitable for direct plotting or already inverse_transformed.
    # If inverse transform is needed:
    # target_scaler = data_processor.get_scaler('target') # You'd need data_processor instance or load scaler

    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            if samples_collected >= n_samples:
                break
            
            # Assuming for sequence models, batch_data is (input_sequence, target_values)
            inputs, true_values_at_vertices = batch_data[0].to(device), batch_data[1].to(device)
            
            predicted_values_at_vertices = model(inputs)

            for j in range(true_values_at_vertices.size(0)): # Iterate through batch
                if samples_collected >= n_samples:
                    break

                true_sample_flat = true_values_at_vertices[j].cpu().numpy()
                predicted_sample_flat = predicted_values_at_vertices[j].cpu().numpy()

                # Ensure values are 1D array of size num_vertices
                if true_sample_flat.ndim > 1: true_sample_flat = true_sample_flat.squeeze()
                if predicted_sample_flat.ndim > 1: predicted_sample_flat = predicted_sample_flat.squeeze()

                if true_sample_flat.shape[0] != vertices.shape[0]:
                    log.error(f"Prediction Sample {samples_collected+1}: Number of true features ({true_sample_flat.shape[0]}) "
                              f"does not match number of vertices ({vertices.shape[0]}). Cannot use tripcolor.")
                    samples_collected += 1
                    continue
                if predicted_sample_flat.shape[0] != vertices.shape[0]:
                    log.error(f"Prediction Sample {samples_collected+1}: Number of predicted features ({predicted_sample_flat.shape[0]}) "
                              f"does not match number of vertices ({vertices.shape[0]}). Cannot use tripcolor.")
                    samples_collected += 1
                    continue

                ax_true = axes[samples_collected, 0]
                ax_pred = axes[samples_collected, 1]

                vmin = min(true_sample_flat.min(), predicted_sample_flat.min())
                vmax = max(true_sample_flat.max(), predicted_sample_flat.max())

                # Plot true values
                tpc_true = ax_true.tripcolor(vertices[:, 0], vertices[:, 1], triangles, true_sample_flat,
                                             cmap='viridis', vmin=vmin, vmax=vmax, shading='gouraud')
                ax_true.set_title(f"True Sample {samples_collected+1}")
                ax_true.set_xlabel("Longitude")
                ax_true.set_ylabel("Latitude")
                ax_true.axis('equal')

                # Plot predicted values
                tpc_pred = ax_pred.tripcolor(vertices[:, 0], vertices[:, 1], triangles, predicted_sample_flat,
                                             cmap='viridis', vmin=vmin, vmax=vmax, shading='gouraud')
                ax_pred.set_title(f"Predicted Sample {samples_collected+1}")
                ax_pred.set_xlabel("Longitude")
                ax_pred.set_ylabel("Latitude")
                ax_pred.axis('equal')
                
                samples_collected += 1

    if samples_collected == 0:
        log.warning("No samples were collected/plotted for prediction. Skipping plot saving.")
        plt.close(fig)
        return

    fig.colorbar(tpc_pred, ax=axes[:, 1], orientation='vertical', fraction=0.046, pad=0.04)
    plt.tight_layout(rect=[0, 0, 0.95, 1])

    if save_dir is None:
        save_dir = os.path.join(cfg.paths.visualization_dir, "predictions")
    os.makedirs(save_dir, exist_ok=True)
    
    plot_filename = os.path.join(save_dir, f"{cfg.experiment.name}_prediction_tripcolor.png")
    plt.savefig(plot_filename)
    log.info(f"Prediction (tripcolor) samples plot saved to {plot_filename}")
    plt.close(fig)

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
    ae_metrics: Optional[Dict[str, Any]],
    lstm_metrics: Optional[Dict[str, Any]],
    cfg: DictConfig,
    save_dir: Optional[str] = None
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
