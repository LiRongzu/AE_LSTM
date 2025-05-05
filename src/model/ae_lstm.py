#!/usr/bin/env python
# src/model/ae_lstm.py - Combined autoencoder and LSTM model

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Union, Any
from omegaconf import DictConfig
import logging

# Import project models
from src.model.autoencoder import AutoencoderModel
from src.model.lstm import LSTMModel

log = logging.getLogger(__name__)

class AELSTMModel(nn.Module):
    """
    Combined autoencoder and LSTM model for spatiotemporal prediction.
    """
    def __init__(
        self,
        autoencoder: nn.Module,
        lstm: nn.Module,
        cfg: DictConfig
    ):
        super(AELSTMModel, self).__init__()
        self.cfg = cfg
        self.autoencoder = autoencoder
        self.lstm = lstm
        
        # Additional parameters
        self.additional_input_features = cfg.model.ae_lstm.get("additional_input_features", 0)
        self.train_ae_end_to_end = cfg.model.ae_lstm.get("train_ae_end_to_end", False)
        
        # Freeze autoencoder if not training end-to-end
        if not self.train_ae_end_to_end:
            for param in self.autoencoder.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through combined AE-LSTM model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        batch_size, seq_len, input_dim = x.shape
        device = x.device
        
        # Split input into main features and additional features if needed
        if self.additional_input_features > 0:
            main_features = x[:, :, :-self.additional_input_features]
            additional_features = x[:, :, -self.additional_input_features:]
        else:
            main_features = x
            additional_features = None
        
        # Encode each time step through autoencoder
        encoded_sequence = []
        
        for t in range(seq_len):
            # Get features for current time step
            current_features = main_features[:, t, :]
            
            # Encode
            latent = self.autoencoder.encode(current_features)
            encoded_sequence.append(latent)
        
        # Stack encoded features
        encoded_sequence = torch.stack(encoded_sequence, dim=1)  # [batch_size, seq_len, latent_dim]
        
        # Combine with additional features if needed
        if additional_features is not None:
            lstm_input = torch.cat([encoded_sequence, additional_features], dim=2)
        else:
            lstm_input = encoded_sequence
        
        # Predict with LSTM
        latent_pred = self.lstm(lstm_input)  # [batch_size, latent_dim]
        
        # Decode prediction
        output = self.autoencoder.decode(latent_pred)  # [batch_size, output_dim]
        
        return output
    
    def predict_sequence(
        self, 
        initial_sequence: torch.Tensor, 
        steps: int
    ) -> torch.Tensor:
        """
        Generate multi-step predictions.
        
        Args:
            initial_sequence: Initial sequence to start prediction
                             [batch_size, seq_len, input_dim]
            steps: Number of steps to predict
            
        Returns:
            Predicted sequence [batch_size, steps, output_dim]
        """
        self.eval()
        device = next(self.parameters()).device
        batch_size = initial_sequence.size(0)
        
        # Split input into main features and additional features if needed
        if self.additional_input_features > 0:
            main_features = initial_sequence[:, :, :-self.additional_input_features]
            additional_features = initial_sequence[:, :, -self.additional_input_features:]
        else:
            main_features = initial_sequence
            additional_features = None
        
        # Encode initial sequence
        encoded_sequence = []
        
        for t in range(main_features.size(1)):
            # Get features for current time step
            current_features = main_features[:, t, :]
            
            # Encode
            latent = self.autoencoder.encode(current_features)
            encoded_sequence.append(latent)
        
        # Stack encoded features
        encoded_sequence = torch.stack(encoded_sequence, dim=1)  # [batch_size, seq_len, latent_dim]
        
        # Combine with additional features if needed
        if additional_features is not None:
            current_lstm_input = torch.cat([encoded_sequence, additional_features], dim=2)
        else:
            current_lstm_input = encoded_sequence
        
        # Initialize predictions
        output_dim = self.autoencoder.input_dim
        predictions = torch.zeros(batch_size, steps, output_dim).to(device)
        
        # Hidden state
        h = None
        
        with torch.no_grad():
            # Make predictions one step at a time
            for i in range(steps):
                # Generate latent prediction
                lstm_output, h = self.lstm.lstm(current_lstm_input, h)
                latent_pred = self.lstm.fc(lstm_output[:, -1, :])
                
                # Decode to output
                output = self.autoencoder.decode(latent_pred)
                
                # Store prediction
                predictions[:, i, :] = output
                
                # Encode prediction for next input
                encoded_pred = self.autoencoder.encode(output).unsqueeze(1)
                
                # Update sequence for next prediction
                if additional_features is not None:
                    # For simplicity, repeat the last additional features
                    # In a real-world scenario, you might want to have future values of additional features
                    next_additional_features = additional_features[:, -1:, :].clone()
                    
                    # Remove first time step and add new prediction
                    new_encoded_sequence = torch.cat([
                        current_lstm_input[:, 1:, :-self.additional_input_features], 
                        encoded_pred
                    ], dim=1)
                    
                    # Combine with additional features
                    current_lstm_input = torch.cat([new_encoded_sequence, next_additional_features], dim=2)
                else:
                    # Remove first time step and add new prediction
                    current_lstm_input = torch.cat([
                        current_lstm_input[:, 1:, :], 
                        encoded_pred
                    ], dim=1)
        
        return predictions


class UNetLSTMModel(nn.Module):
    """
    Combined U-Net autoencoder and LSTM model for spatiotemporal prediction.
    """
    def __init__(
        self,
        autoencoder: nn.Module,
        lstm: nn.Module,
        cfg: DictConfig
    ):
        super(UNetLSTMModel, self).__init__()
        self.cfg = cfg
        self.autoencoder = autoencoder
        self.lstm = lstm
        
        # Additional parameters
        self.train_ae_end_to_end = cfg.model.ae_lstm.get("train_ae_end_to_end", False)
        
        # Freeze autoencoder if not training end-to-end
        if not self.train_ae_end_to_end:
            for param in self.autoencoder.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through combined U-Net LSTM model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, height*width]
            
        Returns:
            Output tensor of shape [batch_size, height*width]
        """
        batch_size, seq_len, input_dim = x.shape
        device = x.device
        
        # Encode each time step through U-Net encoder
        encoded_sequence = []
        
        for t in range(seq_len):
            # Get features for current time step
            current_features = x[:, t, :]
            
            # Encode
            latent = self.autoencoder.encode(current_features)
            encoded_sequence.append(latent)
        
        # Stack encoded features
        encoded_sequence = torch.stack(encoded_sequence, dim=1)  # [batch_size, seq_len, latent_dim]
        
        # Predict with LSTM
        latent_pred = self.lstm(encoded_sequence)  # [batch_size, latent_dim]
        
        # Decode prediction
        output = self.autoencoder.decode(latent_pred)  # [batch_size, height*width]
        
        return output


def get_ae_lstm_model(cfg: DictConfig, autoencoder: nn.Module, lstm: nn.Module) -> nn.Module:
    """
    Factory function to create appropriate AE-LSTM model.
    
    Args:
        cfg: Configuration object
        autoencoder: Pretrained autoencoder model
        lstm: Pretrained LSTM model
    
    Returns:
        AE-LSTM model instance
    """
    ae_type = cfg.model.autoencoder.type.lower()
    
    if ae_type == "unet":
        return UNetLSTMModel(autoencoder, lstm, cfg)
    else:
        return AELSTMModel(autoencoder, lstm, cfg)
