#!/usr/bin/env python
# src/model/ae_lstm.py - Combined autoencoder and LSTM model

from src.model.autoencoder import AutoencoderModel
# from src.model.lstm import LSTMModel # Predictive model is now generic
import torch.nn as nn # For nn.Module typing
from omegaconf import DictConfig
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Union, Any
import logging

log = logging.getLogger(__name__)

class AEPredictiveModel(nn.Module): # Renamed from AELSTMModel
    """
    Combined autoencoder and predictive model (LSTM, Mamba, Transformer) for spatiotemporal prediction.
    """
    def __init__(
        self,
        autoencoder: AutoencoderModel,
        predictive_model: nn.Module, # Changed from lstm_model: LSTMModel
        cfg: DictConfig # cfg might be used for specific behaviors if needed
    ):
        super().__init__()
        self.autoencoder = autoencoder
        self.predictive_model = predictive_model # Changed from self.lstm

        self.autoencoder_input_dim = self.autoencoder.input_dim
        self.autoencoder_latent_dim = self.autoencoder.latent_dim
        log.info(f"AEPredictiveModel initialized. AE input_dim: {self.autoencoder_input_dim}, AE latent_dim: {self.autoencoder_latent_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through combined AE-Predictive model.

        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]

        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        feature_dim_of_x = x.shape[-1]

        if feature_dim_of_x == self.autoencoder_input_dim:
            # Input 'x' is raw features.
            log.debug(f"AEPredictiveModel forward: raw input x shape: {x.shape}")
            batch_size, seq_len, _ = x.shape
            x_reshaped = x.reshape(-1, self.autoencoder_input_dim)
            latent_reshaped = self.autoencoder.encode(x_reshaped)
            latent_sequence = latent_reshaped.reshape(batch_size, seq_len, self.autoencoder_latent_dim)

            if latent_sequence.shape[1] <= 1: # seq_len <=1
                log.warning(f"AEPredictiveModel raw input path: seq_len ({latent_sequence.shape[1]}) <= 1. Using full latent_sequence for predictive model.")
                predictive_input_sequence = latent_sequence
            else:
                # Assuming the predictive model takes all but last step to predict the last one,
                # or that the predictive model handles sequence appropriately.
                # For direct replacement of LSTM logic that predicts next step from sequence:
                predictive_input_sequence = latent_sequence # The predictive model itself should handle seq_len

        elif feature_dim_of_x == self.autoencoder_latent_dim:
            # Input 'x' is already latent codes.
            log.debug(f"AEPredictiveModel forward: latent input x shape: {x.shape}")
            if x.ndim == 2:
                log.debug("AEPredictiveModel forward: latent input x is 2D. Unsqueezing to add seq_len=1.")
                predictive_input_sequence = x.unsqueeze(1)
            elif x.ndim == 3:
                current_seq_len = x.shape[1]
                if current_seq_len == 0:
                    raise ValueError("AEPredictiveModel: Latent input x (3D) has sequence length 0.")
                else: # current_seq_len >= 1
                    log.debug(f"AEPredictiveModel forward: latent input x is 3D with seq_len={current_seq_len}. Using x directly for predictive model.")
                    predictive_input_sequence = x
            else:
                raise ValueError(
                    f"AEPredictiveModel: Latent input x has unexpected ndim {x.ndim}. Expected 2 or 3. Shape: {x.shape}"
                )

        else:
            raise ValueError(
                f"AEPredictiveModel: Input feature dimension {feature_dim_of_x} (shape {x.shape}) is unexpected. "
                f"Expected autoencoder_input_dim ({self.autoencoder_input_dim}) "
                f"or autoencoder_latent_dim ({self.autoencoder_latent_dim})."
            )

        log.debug(f"AEPredictiveModel: predictive_input_sequence shape: {predictive_input_sequence.shape}")
        # The predictive_model (LSTM, Mamba, Transformer) is expected to output the predicted latent code(s).
        # For models that output the last step's prediction (batch, latent_dim):
        predicted_latent_codes = self.predictive_model(predictive_input_sequence)

        # If predictive_model outputs full sequence (batch, seq, latent_dim), take last step:
        if predicted_latent_codes.ndim == 3 and predicted_latent_codes.shape[1] > 1:
             # This depends on predictive_model's design. If it already outputs last step, this is not needed.
             # The current LSTM, Mamba, Transformer are designed to output (batch, latent_dim) or (batch, seq, latent_dim)
             # For now, assume predictive_model's output is appropriate for decoding directly if it's already (B, D)
             # or (B, S, D) where S=1. If S > 1, the original AELSTM took last step from LSTM.
             # Let's assume the individual models (LSTM, Mamba, Transformer) are already configured
             # to output the desired prediction (e.g. last time step's features)
             # If the predictive_model always outputs (Batch, Features) for the last time step, this is fine.
             # If it outputs (Batch, Seq, Features), and we need the last, then:
             # predicted_latent_codes = predicted_latent_codes[:, -1, :] # Uncomment if necessary
             pass


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

        # Hidden state for internal loop, if predictive_model has stateful components (like underlying LSTM cells)
        # This predict_sequence method is quite specific to how an LSTM with explicit hidden state passing would work.
        # Mamba and Transformer typically manage their own state/context internally over the sequence.
        # This method might need significant rework for Mamba/Transformer or be simplified if
        # self.predictive_model.predict_sequence() is expected.
        # For now, let's assume self.predictive_model can be called iteratively.
        # The direct use of self.predictive_model.lstm and .fc below is problematic.

        predictive_model_internal_lstm = None
        if hasattr(self.predictive_model, 'lstm') and hasattr(self.predictive_model, 'fc'):
             # This is an attempt to keep compatibility if predictive_model is the old LSTMModel
             predictive_model_internal_lstm = self.predictive_model.lstm
             predictive_model_internal_fc = self.predictive_model.fc
        else:
            # For Mamba/Transformer, this iterative prediction with explicit state (h) is not standard.
            # They typically take the whole sequence and predict.
            # This iterative prediction logic might be too LSTM-specific.
            # A more generic approach: call self.predictive_model(current_input_sequence)
            # and it should return the next step's prediction.
            log.warning("predict_sequence in AEPredictiveModel might be LSTM-specific and may not work correctly with Mamba/Transformer if they don't have .lstm and .fc attributes or expect iterative state passing.")
            # Fallback: if the model is not an LSTM with .lstm and .fc, this part will likely fail or behave unexpectedly.
            # A proper solution would be to have a predict_one_step method in each predictive model.
            pass

        h = None # Hidden state for LSTM-like models

        with torch.no_grad():
            # Make predictions one step at a time
            for i in range(steps):
                if predictive_model_internal_lstm: # Original LSTM-specific path
                    lstm_output, h = predictive_model_internal_lstm(current_lstm_input, h)
                    latent_pred = predictive_model_internal_fc(lstm_output[:, -1, :])
                else: # Generic path - assumes predictive_model outputs next step directly
                    # This might require current_lstm_input to be of appropriate shape for the model
                    latent_pred = self.predictive_model(current_lstm_input)
                    if latent_pred.ndim == 3 and latent_pred.shape[1] == 1: # Ensure (B, D)
                        latent_pred = latent_pred.squeeze(1)
                    elif latent_pred.ndim == 3 and latent_pred.shape[1] > 1: # If model outputs full seq
                        latent_pred = latent_pred[:, -1, :]


                # Decode to output
                output = self.autoencoder.decode(latent_pred) # latent_pred should be (B, LatentDim)

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
                        current_lstm_input[:, 1:, :], # This assumes current_lstm_input is latent
                        encoded_pred # encoded_pred is (B, 1, LatentDim)
                    ], dim=1)

        return predictions


class UNetPredictiveModel(nn.Module): # Renamed from UNetLSTMModel
    """
    Combined U-Net autoencoder and Predictive model for spatiotemporal prediction.
    """
    def __init__(
        self,
        autoencoder: nn.Module, # Should be the U-Net autoencoder
        predictive_model: nn.Module, # Changed from lstm
        cfg: DictConfig
    ):
        super(UNetPredictiveModel, self).__init__() # Corrected super call
        self.cfg = cfg
        self.autoencoder = autoencoder
        self.predictive_model = predictive_model # Changed from self.lstm

        # Additional parameters
        # Accessing ae_predictive, assuming config structure is updated
        self.train_ae_end_to_end = cfg.model.ae_predictive.get("train_ae_end_to_end", False)

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

        # Predict with Predictive Model
        latent_pred = self.predictive_model(encoded_sequence)  # [batch_size, latent_dim]

        # Decode prediction
        output = self.autoencoder.decode(latent_pred)  # [batch_size, height*width]

        return output


def get_ae_predictive_model(cfg: DictConfig, autoencoder: nn.Module, predictive_model: nn.Module) -> nn.Module: # Renamed
    """
    Factory function to create appropriate AE-Predictive model.

    Args:
        cfg: Configuration object
        autoencoder: Pretrained autoencoder model
        predictive_model: Pretrained predictive model (LSTM, Mamba, etc.)

    Returns:
        AE-Predictive model instance
    """
    ae_type = cfg.model.autoencoder.type.lower()

    if ae_type == "unet":
        return UNetPredictiveModel(autoencoder, predictive_model, cfg) # Updated class name
    else:
        return AEPredictiveModel(autoencoder, predictive_model, cfg) # Updated class name
