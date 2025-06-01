#!/usr/bin/env python
# src/model/autoencoder.py - Autoencoder model implementations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Union, Any
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)

class Encoder(nn.Module):
    """
    Encoder network for the autoencoder.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_layers: List[int],
        activation: str = "ReLU",
        dropout_rate: float = 0.2
    ):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build encoder layers
        layers = []
        input_size = input_dim
        
        # Create activation function
        if activation == "ReLU":
            act_fn = nn.ReLU()
        elif activation == "LeakyReLU":
            act_fn = nn.LeakyReLU(0.2)
        elif activation == "Tanh":
            act_fn = nn.Tanh()
        elif activation == "Sigmoid":
            act_fn = nn.Sigmoid()
        else:
            act_fn = nn.ReLU()
            log.warning(f"Unknown activation {activation}, using ReLU")
            
        # Create hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_dim))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_dim
            
        # Final layer to latent space
        layers.append(nn.Linear(input_size, latent_dim))
        
        # Create sequential model
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder"""
        return self.encoder(x)


class Decoder(nn.Module):
    """
    Decoder network for the autoencoder.
    """
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_layers: List[int],
        activation: str = "ReLU",
        dropout_rate: float = 0.2
    ):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Build decoder layers (reverse of encoder)
        layers = []
        input_size = latent_dim
        
        # Create activation function
        if activation == "ReLU":
            act_fn = nn.ReLU()
        elif activation == "LeakyReLU":
            act_fn = nn.LeakyReLU(0.2)
        elif activation == "Tanh":
            act_fn = nn.Tanh()
        elif activation == "Sigmoid":
            act_fn = nn.Sigmoid()
        else:
            act_fn = nn.ReLU()
            
        # Create hidden layers in reverse order
        hidden_layers_reversed = list(reversed(hidden_layers))
        for hidden_dim in hidden_layers_reversed:
            layers.append(nn.Linear(input_size, hidden_dim))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_dim
            
        # Final layer to output dimension
        layers.append(nn.Linear(input_size, output_dim))
        
        # Create sequential model
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder"""
        return self.decoder(z)


class AutoencoderModel(nn.Module):
    """
    Standard autoencoder model combining encoder and decoder.
    """
    def __init__(self, ae_cfg: DictConfig): # Changed signature
        super(AutoencoderModel, self).__init__()
        self.cfg = ae_cfg # Storing it as self.cfg for minimal internal change, but it's the ae_cfg
        self.input_dim = ae_cfg.input_dim
        self.latent_dim = ae_cfg.latent_dim
        self.hidden_layers = list(ae_cfg.hidden_layers) # Ensure it's a list
        self.activation = ae_cfg.activation
        self.dropout_rate = ae_cfg.dropout_rate
        
        # Create encoder and decoder
        self.encoder = Encoder(
            self.input_dim,
            self.latent_dim,
            self.hidden_layers,
            self.activation,
            self.dropout_rate
        )
        
        self.decoder = Decoder(
            self.latent_dim,
            self.input_dim,
            self.hidden_layers,
            self.activation,
            self.dropout_rate
        )
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space"""
        return self.encoder(x)
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent space to output"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass through autoencoder"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon


class UNetAutoencoder(nn.Module):
    """
    U-Net style autoencoder model for spatial data.
    """
    def __init__(self, ae_cfg: DictConfig): # Changed signature
        super(UNetAutoencoder, self).__init__()
        self.cfg = ae_cfg # Storing it as self.cfg
        self.input_dim = ae_cfg.input_dim
        self.latent_dim = ae_cfg.latent_dim
        
        # Calculate spatial dimensions assuming the input is a square grid
        # This is an example for a 2D grid
        self.grid_size = int(np.sqrt(self.input_dim))
        
        # Encoder (downsampling)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Latent space (adjust size based on downsampling)
        ds_factor = 8  # 3 max pooling layers with stride 2
        self.latent_size = self.grid_size // ds_factor
        self.fc_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * self.latent_size * self.latent_size, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 64 * self.latent_size * self.latent_size)
        )
        
        # Decoder (upsampling)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.ReLU()
        )
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space"""
        # Reshape input to 2D grid if needed
        batch_size = x.size(0)
        if len(x.shape) == 2:  # [batch_size, input_dim]
            x = x.view(batch_size, 1, self.grid_size, self.grid_size)
            
        # Encoding
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # To latent space
        latent = self.fc_latent(e3)[:, :self.latent_dim]
        
        return latent
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent space to output"""
        batch_size = z.size(0)
        
        # From latent space
        latent_expanded = F.relu(self.fc_latent[:self.latent_dim, :](z))
        d3 = latent_expanded.view(batch_size, 64, self.latent_size, self.latent_size)
        
        # Decoding
        d2 = self.dec3(d3)
        d1 = self.dec2(d2)
        out = self.dec1(d1)
        
        # Reshape output if needed
        if len(out.shape) > 2:  # [batch_size, 1, height, width]
            out = out.view(batch_size, -1)
            
        return out
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass through autoencoder"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon


class MaskedAutoencoder(AutoencoderModel):
    """
    Masked autoencoder that can handle missing data.
    """
    def __init__(self, ae_cfg: DictConfig): # Changed signature
        super(MaskedAutoencoder, self).__init__(ae_cfg) # Pass ae_cfg to parent
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional mask.
        
        Args:
            x: Input tensor
            mask: Binary mask (1 for valid data, 0 for missing)
        
        Returns:
            Reconstructed output
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        
        # Apply mask if provided
        if mask is not None:
            x_recon = x_recon * mask
            
        return x_recon


def get_autoencoder_model(ae_cfg: DictConfig) -> nn.Module: # Changed signature
    """
    Factory function to create appropriate autoencoder model.
    
    Args:
        ae_cfg: Autoencoder-specific configuration object
    
    Returns:
        Autoencoder model instance
    """
    ae_type = ae_cfg.type.lower() # Access type directly from ae_cfg
    log.info(f"Creating autoencoder of type: {ae_type} with input_dim: {ae_cfg.input_dim}") # Log using ae_cfg
    
    if ae_type == "standard":
        return AutoencoderModel(ae_cfg) # Pass ae_cfg
    elif ae_type == "unet":
        return UNetAutoencoder(ae_cfg) # Pass ae_cfg
    elif ae_type == "masked":
        return MaskedAutoencoder(ae_cfg) # Pass ae_cfg
    else:
        log.warning(f"Unknown autoencoder type: {ae_type}, using standard")
        return AutoencoderModel(ae_cfg) # Pass ae_cfg
