#!/usr/bin/env python
# src/model/ae_lstm.py - Combined autoencoder and LSTM model

from src.model.autoencoder import AutoencoderModel
from src.model.lstm import LSTMModel
from omegaconf import DictConfig
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Union, Any
import logging

log = logging.getLogger(__name__)

class AELSTMModel(nn.Module):
    """
    Combined autoencoder and LSTM model for spatiotemporal prediction.
    """
    def __init__(
        self,
        autoencoder: AutoencoderModel,
        lstm_model: LSTMModel,
        cfg: DictConfig
    ):
        super().__init__()
        self.autoencoder = autoencoder
        self.lstm = lstm_model
        
        self.autoencoder_input_dim = self.autoencoder.input_dim
        self.autoencoder_latent_dim = self.autoencoder.latent_dim
        log.info(f"AELSTMModel initialized. AE input_dim: {self.autoencoder_input_dim}, AE latent_dim: {self.autoencoder_latent_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through combined AE-LSTM model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        feature_dim_of_x = x.shape[-1]
        
        if feature_dim_of_x == self.autoencoder_input_dim:
            # Input 'x' is raw features.
            log.debug(f"AELSTMModel forward: raw input x shape: {x.shape}")
            batch_size, seq_len, _ = x.shape
            x_reshaped = x.reshape(-1, self.autoencoder_input_dim)
            latent_reshaped = self.autoencoder.encode(x_reshaped)
            latent_sequence = latent_reshaped.reshape(batch_size, seq_len, self.autoencoder_latent_dim)
            
            if latent_sequence.shape[1] <= 1: # seq_len <=1
                log.warning(f"AELSTMModel raw input path: seq_len ({latent_sequence.shape[1]}) <= 1. Using full latent_sequence for LSTM.")
                lstm_input_sequence = latent_sequence
            else:
                lstm_input_sequence = latent_sequence[:, :-1, :]
            
        elif feature_dim_of_x == self.autoencoder_latent_dim:
            # Input 'x' is already latent codes.
            log.debug(f"AELSTMModel forward: latent input x shape: {x.shape}")
            if x.ndim == 2:
                log.debug("AELSTMModel forward: latent input x is 2D. Unsqueezing to add seq_len=1.")
                lstm_input_sequence = x.unsqueeze(1)
            elif x.ndim == 3:
                current_seq_len = x.shape[1]
                if current_seq_len == 0:
                    raise ValueError("AELSTMModel: Latent input x (3D) has sequence length 0.")
                elif current_seq_len == 1:
                    log.debug("AELSTMModel forward: latent input x is 3D with seq_len=1. Using x directly for LSTM.")
                    lstm_input_sequence = x
                else: # current_seq_len > 1
                    log.debug(f"AELSTMModel forward: latent input x is 3D with seq_len={current_seq_len}. Using x[:, :-1, :] for LSTM.")
                    lstm_input_sequence = x[:, :-1, :]
            else:
                raise ValueError(
                    f"AELSTMModel: Latent input x has unexpected ndim {x.ndim}. Expected 2 or 3. Shape: {x.shape}"
                )
            
        else:
            raise ValueError(
                f"AELSTMModel: Input feature dimension {feature_dim_of_x} (shape {x.shape}) is unexpected. "
                f"Expected autoencoder_input_dim ({self.autoencoder_input_dim}) "
                f"or autoencoder_latent_dim ({self.autoencoder_latent_dim})."
            )
            
        log.debug(f"AELSTMModel: lstm_input_sequence shape: {lstm_input_sequence.shape}")
        predicted_latent_codes = self.lstm(lstm_input_sequence)
        # The output of self.lstm (LSTMModel) should be the predicted latent code(s).
        # Shape could be (batch_size, latent_dim) or (batch_size, some_seq_len, latent_dim)
        # depending on LSTMModel's fc layer and predict_last_only config.

        # Decode the predicted latent codes to get final predictions in original feature space.
        final_predictions = self.autoencoder.decode(predicted_latent_codes)
        
        return final_predictions
    
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
