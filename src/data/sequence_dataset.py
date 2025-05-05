#!/usr/bin/env python
# src/data/sequence_dataset.py - Dataset class for sequences

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Any, Optional, Union
import logging

log = logging.getLogger(__name__)

class SequenceDataset(Dataset):
    """
    Dataset for time series sequence data.
    
    This dataset creates input-target pairs for sequence prediction tasks.
    The input is a sequence of `sequence_length` timesteps, and the target
    is the `forecast_horizon` timesteps ahead.
    """
    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int,
        forecast_horizon: int = 1,
        target_idx: Optional[int] = None,
        transform=None,
        target_transform=None
    ):
        """
        Initialize sequence dataset.
        
        Args:
            data: Time series data with shape [timesteps, features] or [timesteps, features, ...]
            sequence_length: Length of input sequence
            forecast_horizon: How many steps into the future to predict
            target_idx: If not None, use only the specified feature index for the target
            transform: Transform to apply on input sequences
            target_transform: Transform to apply on target sequences
        """
        self.data = data
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.target_idx = target_idx
        self.transform = transform
        self.target_transform = target_transform
        
        # Store data shape
        self.n_timesteps = data.shape[0]
        self.features_shape = data.shape[1:] if len(data.shape) > 1 else (1,)
        
        # Determine number of valid sequences
        self.n_sequences = max(0, self.n_timesteps - self.sequence_length - self.forecast_horizon + 1)
        
        if self.n_sequences <= 0:
            log.warning(f"Not enough timesteps ({self.n_timesteps}) for sequence length "
                        f"({self.sequence_length}) and horizon ({self.forecast_horizon})")
        
    def __len__(self) -> int:
        return self.n_sequences
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence pair (input, target) by index.
        
        Args:
            idx: Sequence index
            
        Returns:
            Tuple containing input sequence and target
        """
        if idx < 0 or idx >= self.n_sequences:
            raise IndexError(f"Index {idx} out of range for dataset with {self.n_sequences} sequences")
        
        # Get input sequence
        input_start = idx
        input_end = idx + self.sequence_length
        input_seq = self.data[input_start:input_end]
        
        # Get target - last input + forecast_horizon
        target_idx = input_end + self.forecast_horizon - 1
        target = self.data[target_idx]
        
        # Extract specific feature for target if specified
        if self.target_idx is not None:
            if isinstance(self.target_idx, (list, tuple)):
                target = target[self.target_idx]
            else:
                target = target[self.target_idx:self.target_idx+1]
        
        # Convert to tensors
        input_seq = torch.FloatTensor(input_seq)
        target = torch.FloatTensor(target)
        
        # Apply transforms if provided
        if self.transform:
            input_seq = self.transform(input_seq)
        if self.target_transform:
            target = self.target_transform(target)
            
        return input_seq, target
        
class SpatioTemporalSequenceDataset(SequenceDataset):
    """
    Dataset for spatiotemporal sequence data.
    
    This is a specialized version of SequenceDataset for handling
    spatiotemporal data like gridded salinity fields.
    """
    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int,
        forecast_horizon: int = 1,
        external_factors: Optional[np.ndarray] = None,
        transform=None,
        target_transform=None
    ):
        """
        Initialize spatiotemporal sequence dataset.
        
        Args:
            data: Spatiotemporal data with shape [timesteps, height, width] or similar
            sequence_length: Length of input sequence
            forecast_horizon: How many steps into the future to predict
            external_factors: Optional external factors with shape [timesteps, n_factors]
            transform: Transform to apply on input sequences
            target_transform: Transform to apply on target sequences
        """
        super().__init__(
            data=data,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
            transform=transform,
            target_transform=target_transform
        )
        
        self.external_factors = external_factors
        if external_factors is not None:
            assert external_factors.shape[0] >= self.n_timesteps, \
                f"External factors length ({external_factors.shape[0]}) must match or exceed " \
                f"data timepoints ({self.n_timesteps})"
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence pair (input, target) by index.
        
        Args:
            idx: Sequence index
            
        Returns:
            Tuple containing input sequence and target
        """
        # Get base sequence and target
        input_seq, target = super().__getitem__(idx)
        
        # Add external factors if available
        if self.external_factors is not None:
            # Get corresponding external factors
            input_start = idx
            input_end = idx + self.sequence_length
            ext_factors = self.external_factors[input_start:input_end]
            ext_factors = torch.FloatTensor(ext_factors)
            
            # Return input as tuple of (sequence, external_factors)
            return (input_seq, ext_factors), target
            
        return input_seq, target


class LatentSequenceDataset(Dataset):
    """
    Dataset for sequences of latent representations.
    
    This dataset handles sequences of latent vectors produced by an autoencoder
    for training temporal models like LSTMs.
    """
    def __init__(
        self,
        latent_data: np.ndarray,
        sequence_length: int,
        forecast_horizon: int = 1,
        external_factors: Optional[np.ndarray] = None,
        transform=None,
        target_transform=None
    ):
        """
        Initialize latent sequence dataset.
        
        Args:
            latent_data: Latent representations with shape [timesteps, latent_dim]
            sequence_length: Length of input sequence
            forecast_horizon: How many steps into the future to predict
            external_factors: Optional external factors with shape [timesteps, n_factors]
            transform: Transform to apply on input sequences
            target_transform: Transform to apply on target sequences
        """
        self.latent_data = latent_data
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.external_factors = external_factors
        self.transform = transform
        self.target_transform = target_transform
        
        # Store data shape
        self.n_timesteps = latent_data.shape[0]
        self.latent_dim = latent_data.shape[1]
        
        # Determine number of valid sequences
        self.n_sequences = max(0, self.n_timesteps - self.sequence_length - self.forecast_horizon + 1)
        
        if self.n_sequences <= 0:
            log.warning(f"Not enough timesteps ({self.n_timesteps}) for sequence length "
                        f"({self.sequence_length}) and horizon ({self.forecast_horizon})")
                        
        if external_factors is not None:
            assert external_factors.shape[0] >= self.n_timesteps, \
                f"External factors length ({external_factors.shape[0]}) must match or exceed " \
                f"data timepoints ({self.n_timesteps})"
    
    def __len__(self) -> int:
        return self.n_sequences
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence pair (input, target) by index.
        
        Args:
            idx: Sequence index
            
        Returns:
            Tuple containing input sequence and target
        """
        if idx < 0 or idx >= self.n_sequences:
            raise IndexError(f"Index {idx} out of range for dataset with {self.n_sequences} sequences")
        
        # Get input sequence
        input_start = idx
        input_end = idx + self.sequence_length
        input_seq = self.latent_data[input_start:input_end]
        
        # Get target - last input + forecast_horizon
        target_idx = input_end + self.forecast_horizon - 1
        target = self.latent_data[target_idx]
        
        # Convert to tensors
        input_seq = torch.FloatTensor(input_seq)
        target = torch.FloatTensor(target)
        
        # Add external factors if available
        if self.external_factors is not None:
            # Get corresponding external factors
            ext_factors = self.external_factors[input_start:input_end]
            ext_factors = torch.FloatTensor(ext_factors)
            
            # Combine with input
            input_seq = torch.cat([input_seq, ext_factors], dim=1)
        
        # Apply transforms if provided
        if self.transform:
            input_seq = self.transform(input_seq)
        if self.target_transform:
            target = self.target_transform(target)
            
        return input_seq, target
