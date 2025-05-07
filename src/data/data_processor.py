#!/usr/bin/env python
# src/data/data_processor.py - Data preprocessing and processing utilities

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from omegaconf import DictConfig
import os
import joblib

log = logging.getLogger(__name__)

class DataProcessor:
    """
    Class for processing and preparing data for models.
    """
    def __init__(self, cfg: DictConfig):
        """
        Initialize data processor.
        
        Args:
            cfg: Configuration object
        """
        self.cfg = cfg
        self.target_field = cfg.data.dataset.target_field
        
        # Split settings
        if cfg.data.dataset.split_num:
            self.train_num = 1827
            self.val_num = 365
            self.test_num = 365
        else:
            self.train_ratio = cfg.data.dataset.split_ratio.train
            self.val_ratio = cfg.data.dataset.split_ratio.val
            self.test_ratio = cfg.data.dataset.split_ratio.test

        # Preprocessing settings
        self.standardize = cfg.data.preprocessing.standardization
        self.normalize = cfg.data.preprocessing.normalize
        self.apply_mask = cfg.data.preprocessing.apply_mask
        
        # Sequence settings
        self.seq_length = cfg.data.sequence.sequence_length
        self.pred_horizon = cfg.data.sequence.prediction_horizon
        self.stride = cfg.data.sequence.stride
        self.include_covariates = cfg.data.sequence.include_covariates
        
        # Batch sizes
        self.batch_sizes = {
            'train': cfg.data.batch_size.train,
            'val': cfg.data.batch_size.val,
            'test': cfg.data.batch_size.test
        }
        
        # Initialize scalers
        self.scalers = {}
        
    def process(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Process raw data.
        
        Args:
            data: Dictionary of raw data arrays
            
        Returns:
            Dictionary of processed data
        """
        processed_data = {}
        
        # Get target data
        target_data = data[self.target_field]
        log.info(f"Processing {self.target_field} data with shape {target_data.shape}")
        
        # Reshape data if needed
        if len(target_data.shape) > 2:
            # Assume shape is [time, height, width] or similar
            time_steps, height, width = target_data.shape
            target_data = target_data.reshape(time_steps, -1)
            log.info(f"Reshaped target data to {target_data.shape}")
            
            # Store spatial dimensions for later reconstruction
            processed_data['spatial_dims'] = (height, width)
            
        # Apply mask if provided
        if self.apply_mask and 'mask' in data and data['mask'] is not None:
            mask = data['mask']
            if len(mask.shape) > 1:
                # Flatten mask to match flattened data
                mask = mask.reshape(-1)
                
            # Expand mask to all time steps
            expanded_mask = np.ones(target_data.shape)
            for t in range(target_data.shape[0]):
                expanded_mask[t] = mask
                
            # Apply mask (NaN or zero values where mask is 0)
            target_data = target_data * expanded_mask
            log.info(f"Applied mask to target data")
            
            # Store mask for later use
            processed_data['mask'] = mask
        
        if self.standardize:
            train_end_idx = 0
            if self.cfg.data.dataset.get('split_num', False):
                train_end_idx = self.cfg.data.dataset.train_num
            else:
                train_end_idx = int(self.train_ratio * target_data.shape[0])

            training_slice_for_scaler = target_data[:train_end_idx, :]
            
            if training_slice_for_scaler.size == 0:
                log.error("Training slice for scaler is empty! Cannot fit scaler.")
                # 根据你的逻辑，这里可能需要抛出异常或采取其他错误处理
                # 为简化，我们假设 training_slice_for_scaler 不会为空
            else:
                # Reshape training data for global scaling: (n_samples * n_features, 1)
                original_train_shape = training_slice_for_scaler.shape
                reshaped_train_data = training_slice_for_scaler.reshape(-1, 1)
                log.info(f"Reshaped training data from {original_train_shape} to {reshaped_train_data.shape} for global StandardScaler fitting.")

                scaler = StandardScaler()
                scaler.fit(reshaped_train_data)

                # Log global scaler parameters (they are now single values)
                log.info(f"Global Target StandardScaler - Mean: {scaler.mean_[0]}")
                log.info(f"Global Target StandardScaler - Scale (std): {scaler.scale_[0]}")
                if scaler.scale_[0] == 0:
                    log.warning("Global Target StandardScaler has a zero value in scale_ (std_dev)!")
                
                # Transform the entire target_data globally
                original_full_data_shape = target_data.shape
                reshaped_full_data = target_data.reshape(-1, 1)
                
                transformed_data_1d = scaler.transform(reshaped_full_data)
                
                # Reshape back to original 2D shape
                target_data = transformed_data_1d.reshape(original_full_data_shape)
                
                self.scalers['target'] = scaler
                log.info(f"Globally standardized target data. Final shape: {target_data.shape}")
            
        elif self.normalize:
            train_end_idx = 0
            if self.cfg.data.dataset.get('split_num', False):
                train_end_idx = self.cfg.data.dataset.train_num
            else:
                train_end_idx = int(self.train_ratio * target_data.shape[0])

            training_slice_for_scaler = target_data[:train_end_idx, :]

            if training_slice_for_scaler.size == 0:
                log.error("Training slice for scaler is empty! Cannot fit scaler.")
            else:
                # Reshape training data for global scaling: (n_samples * n_features, 1)
                original_train_shape = training_slice_for_scaler.shape
                reshaped_train_data = training_slice_for_scaler.reshape(-1, 1)
                log.info(f"Reshaped training data from {original_train_shape} to {reshaped_train_data.shape} for global MinMaxScaler fitting.")

                scaler = MinMaxScaler() # 你可以选择指定 feature_range, e.g., feature_range=(0, 1)
                scaler.fit(reshaped_train_data)

                # Log global scaler parameters
                log.info(f"Global Target MinMaxScaler - Min: {scaler.min_[0]} (Actual value used for scaling)") # scaler.min_ is (X_min - data_min_) * scale_ + feature_range_min
                log.info(f"Global Target MinMaxScaler - Scale (1/(data_max-data_min) adjusted for feature_range): {scaler.scale_[0]}")
                log.info(f"Global Target MinMaxScaler - Data Min (learned from data): {scaler.data_min_[0]}")
                log.info(f"Global Target MinMaxScaler - Data Max (learned from data): {scaler.data_max_[0]}")
                
                global_data_range = scaler.data_max_[0] - scaler.data_min_[0]
                if global_data_range == 0:
                    log.warning("Global Target MinMaxScaler has zero range (data_max == data_min)!")

                # Transform the entire target_data globally
                original_full_data_shape = target_data.shape
                reshaped_full_data = target_data.reshape(-1, 1)
                
                transformed_data_1d = scaler.transform(reshaped_full_data)
                
                # Reshape back to original 2D shape
                target_data = transformed_data_1d.reshape(original_full_data_shape)
                
                self.scalers['target'] = scaler
                log.info(f"Globally normalized target data. Final shape: {target_data.shape}")
            
        processed_data['target'] = target_data


        # Process covariates individually
        if self.include_covariates:
            # covariates = [] # Removed: Store individually instead of stacking here
            for field in self.cfg.data.sequence.covariate_fields:
                if field in data:
                    covariate_data = data[field]
                    original_shape = covariate_data.shape
                    log.info(f"Processing covariate {field} with shape {original_shape}")

                    processed_field_data = covariate_data # Start with original data

                    # --- Special handling for wind field ---
                    # Check if it's the wind field and has 4 dimensions (T, C, H, W)
                    if field == 'wind' and len(original_shape) == 4:
                        log.info(f"Keeping spatial dimensions for {field}.")
                        # Scaling needs careful handling for 4D data
                        if self.standardize or self.normalize:
                            # Reshape to (T, C*H*W) for scaler compatibility
                            reshaped_for_scaling = covariate_data.reshape(original_shape[0], -1)
                            log.info(f"Temporarily reshaping {field} to {reshaped_for_scaling.shape} for scaling")

                            # Determine split point for fitting scaler
                            if self.cfg.data.dataset.get('split_num', False):
                                train_end = self.cfg.data.dataset.train_num
                            else:
                                train_end = int(self.train_ratio * original_shape[0])

                            if self.standardize:
                                scaler = StandardScaler()
                            else: # self.normalize
                                scaler = MinMaxScaler()

                            # Fit scaler only on the training part of the reshaped data
                            scaler.fit(reshaped_for_scaling[:train_end])
                            # Transform the entire reshaped data
                            scaled_reshaped = scaler.transform(reshaped_for_scaling)
                            # Reshape back to the original 4D shape
                            processed_field_data = scaled_reshaped.reshape(original_shape)
                            self.scalers[field] = scaler
                            log.info(f"Scaled {field} and reshaped back to {processed_field_data.shape}")
                        # If no scaling, processed_field_data remains the original covariate_data

                    # --- Handling for other covariates (like runoff) ---
                    elif field != 'wind': # Process other fields normally
                        # Reshape if needed (e.g., if some other covariate was > 2D and needs flattening)
                        if len(original_shape) > 2:
                            processed_field_data = covariate_data.reshape(original_shape[0], -1)
                            log.info(f"Reshaped {field} to {processed_field_data.shape}")

                        # Apply scaling if enabled
                        if self.standardize or self.normalize:
                            # Determine split point for fitting scaler
                            if self.cfg.data.dataset.get('split_num', False):
                                train_end = self.cfg.data.dataset.train_num
                            else:
                                train_end = int(self.train_ratio * original_shape[0])

                            if self.standardize:
                                scaler = StandardScaler()
                            else: # self.normalize
                                scaler = MinMaxScaler()

                            # Scaler works directly on 2D data (or already reshaped data)
                            scaler.fit(processed_field_data[:train_end])
                            processed_field_data = scaler.transform(processed_field_data)
                            self.scalers[field] = scaler
                            log.info(f"Scaled {field}")

                    # Store the processed data for this specific field
                    processed_data[field] = processed_field_data
                    log.info(f"Stored processed {field} with final shape {processed_field_data.shape}")

            # Removed: Combining covariates - will be handled later
            # if covariates:
            #     combined_covariates = np.hstack(covariates)
            #     log.info(f"Combined covariates shape: {combined_covariates.shape}")
            #     processed_data['covariates'] = combined_covariates

        # Split data
        time_steps = target_data.shape[0]
        # Determine split points (using ratio or fixed numbers)
        if self.cfg.data.dataset.get('split_num', False): # Check if split_num exists and is True
            train_end = self.cfg.data.dataset.train_num
            val_end = train_end + self.cfg.data.dataset.val_num
            # Ensure test split covers the rest, handle potential mismatch with total time_steps
            test_end = min(val_end + self.cfg.data.dataset.test_num, time_steps)
            log.info(f"Using fixed split numbers: Train={train_end}, Val={self.cfg.data.dataset.val_num}, Test={test_end - val_end}")
        else:
            # Ensure ratios are defined if split_num is false
            if not hasattr(self.cfg.data.dataset, 'split_ratio'):
                 raise ValueError("Split ratios (train, val, test) must be defined in config if split_num is false.")
            train_end = int(self.train_ratio * time_steps)
            val_end = int((self.train_ratio + self.val_ratio) * time_steps)
            test_end = time_steps # Use the rest for test
            log.info(f"Using split ratios: Train={self.train_ratio}, Val={self.val_ratio}, Test={self.test_ratio}")


        splits = {
            'train': slice(0, train_end),
            'val': slice(train_end, val_end),
            'test': slice(val_end, test_end) # Use test_end calculated above
        }


        # Split target and individual covariates
        for split_name, split_slice in splits.items():
            processed_data[f'{split_name}_target'] = target_data[split_slice]
            log.info(f"{split_name} target shape: {processed_data[f'{split_name}_target'].shape}")
            # Split each processed covariate field
            if self.include_covariates:
                for field in self.cfg.data.sequence.covariate_fields:
                    if field in processed_data: # Check if the covariate was successfully processed
                        processed_data[f'{split_name}_{field}'] = processed_data[field][split_slice]
                        log.info(f"{split_name} {field} shape: {processed_data[f'{split_name}_{field}'].shape}")

        # Add checks for val_target before returning processed_data
        if 'val_target' in processed_data:
            val_target_np = processed_data['val_target']
            if val_target_np.size > 0: # Proceed only if not empty
                if np.any(np.isinf(val_target_np)):
                    inf_count = np.sum(np.isinf(val_target_np))
                    log.warning(f"Infinity found in 'val_target' (shape: {val_target_np.shape}) after scaling and splitting. Count: {inf_count}")
                    # Example: log indices of first few inf values
                    inf_indices = np.where(np.isinf(val_target_np))
                    log.warning(f"First few inf indices in val_target: {tuple(idx[:5] for idx in inf_indices)}")
                if np.any(np.isnan(val_target_np)):
                    nan_count = np.sum(np.isnan(val_target_np))
                    log.warning(f"NaN found in 'val_target' (shape: {val_target_np.shape}) after scaling and splitting. Count: {nan_count}")
                    # Example: log indices of first few NaN values
                    nan_indices = np.where(np.isnan(val_target_np))
                    log.warning(f"First few NaN indices in val_target: {tuple(idx[:5] for idx in nan_indices)}")
                
                # Log min/max/mean to see if values are extreme, ignoring NaNs for these stats
                log.info(f"Stats for 'val_target': Min={np.nanmin(val_target_np)}, Max={np.nanmax(val_target_np)}, Mean={np.nanmean(val_target_np)}")
            else:
                log.warning("'val_target' is empty after splitting.")


        # Remove full covariate fields after splitting to keep dict clean
        if self.include_covariates:
            fields_to_remove = [field for field in self.cfg.data.sequence.covariate_fields if field in processed_data]
            for field in fields_to_remove:
                 del processed_data[field]

        # Save scalers
        self._save_scalers()
        
        return processed_data
    
    def _save_scalers(self) -> None:
        """
        Save fitted scalers to disk.
        """
        scaler_dir = os.path.join(self.cfg.paths.output_dir, 'scalers')
        os.makedirs(scaler_dir, exist_ok=True)
        
        for name, scaler in self.scalers.items():
            scaler_path = os.path.join(scaler_dir, f"{name}_scaler.pkl")
            joblib.dump(scaler, scaler_path)
            log.info(f"Saved {name} scaler to {scaler_path}")
    
    def create_ae_datasets(
        self, 
        processed_data: Dict[str, Any]
    ) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
        """
        Create datasets for autoencoder training and testing.
        
        Args:
            processed_data: Dictionary of processed data
            
        Returns:
            Train, validation, and test datasets for autoencoder
        """
        # Get data
        train_data = processed_data['train_target']
        val_data = processed_data['val_target']
        test_data = processed_data['test_target']
        
        # Convert to tensors
        train_tensor = torch.FloatTensor(train_data)
        val_tensor = torch.FloatTensor(val_data)
        test_tensor = torch.FloatTensor(test_data)
        
        # Create datasets
        train_dataset = TensorDataset(train_tensor)
        val_dataset = TensorDataset(val_tensor)
        test_dataset = TensorDataset(test_tensor)
        
        log.info(f"Created autoencoder datasets - Train: {len(train_dataset)}, "
                 f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
                 
        return train_dataset, val_dataset, test_dataset
    
    def generate_latent_representations(
        self,
        autoencoder: torch.nn.Module,
        train_dataset: TensorDataset,
        val_dataset: TensorDataset,
        test_dataset: TensorDataset,
        device: torch.device
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate latent representations using trained autoencoder.
        
        Args:
            autoencoder: Trained autoencoder model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Testing dataset
            device: Device to run inference on
            
        Returns:
            Latent representations for train, val, and test sets
        """
        autoencoder.eval()
        
        # Function to encode a dataset
        def encode_dataset(dataset):
            data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
            encoded_data = []
            
            with torch.no_grad():
                for batch in data_loader:
                    inputs = batch[0].to(device)
                    latent = autoencoder.encode(inputs)
                    encoded_data.append(latent.cpu().numpy())
                    
            return np.vstack(encoded_data)
        
        # Generate latent codes
        train_latent = encode_dataset(train_dataset)
        val_latent = encode_dataset(val_dataset)
        test_latent = encode_dataset(test_dataset)
        
        log.info(f"Generated latent representations - Train: {train_latent.shape}, "
                 f"Val: {val_latent.shape}, Test: {test_latent.shape}")
                 
        return train_latent, val_latent, test_latent
    
    def create_sequences(
        self, 
        data: np.ndarray, 
        covariates: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input/output sequences for time series prediction.
        
        Args:
            data: Time series data [time_steps, features]
            covariates: Optional covariate data [time_steps, covariate_features]
            
        Returns:
            Tuple of input sequences and target sequences
        """
        n_samples = data.shape[0]
        n_features = data.shape[1]
        
        # Calculate number of sequences
        n_sequences = max(0, n_samples - self.seq_length - self.pred_horizon + 1)
        
        # Initialize arrays
        x = np.zeros((n_sequences, self.seq_length, n_features))
        y = np.zeros((n_sequences, n_features))
        
        # Create sequences with stride
        seq_indices = list(range(0, n_sequences, self.stride))
        
        for i, start_idx in enumerate(seq_indices):
            # Input sequence
            x_end = start_idx + self.seq_length
            x[i] = data[start_idx:x_end]
            
            # Target (future value)
            y_idx = x_end + self.pred_horizon - 1
            if y_idx < n_samples:
                y[i] = data[y_idx]
                
        # Include covariates if provided
        if covariates is not None:
            n_covariates = covariates.shape[1]
            x_with_covariates = np.zeros((len(seq_indices), self.seq_length, n_features + n_covariates))
            
            for i, start_idx in enumerate(seq_indices):
                # Input sequence with covariates
                x_end = start_idx + self.seq_length
                
                for t in range(self.seq_length):
                    seq_idx = start_idx + t
                    x_with_covariates[i, t, :n_features] = data[seq_idx]
                    x_with_covariates[i, t, n_features:] = covariates[seq_idx]
                    
            x = x_with_covariates
            
        return x[:len(seq_indices)], y[:len(seq_indices)]
    
    def create_sequence_datasets(
        self,
        train_latent: np.ndarray,
        val_latent: np.ndarray,
        test_latent: np.ndarray,
        processed_data: Dict[str, Any]
    ) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
        """
        Create sequence datasets for LSTM training and testing.
        
        Args:
            train_latent: Training latent representations
            val_latent: Validation latent representations
            test_latent: Testing latent representations
            processed_data: Dictionary of processed data
            
        Returns:
            Train, validation, and test sequence datasets
        """
        # Get covariates if available
        train_covs = processed_data.get('train_covariates', None)
        val_covs = processed_data.get('val_covariates', None)
        test_covs = processed_data.get('test_covariates', None)
        
        # Create sequences
        train_x, train_y = self.create_sequences(train_latent, train_covs)
        val_x, val_y = self.create_sequences(val_latent, val_covs)
        test_x, test_y = self.create_sequences(test_latent, test_covs)
        
        # Convert to tensors
        train_x_tensor = torch.FloatTensor(train_x)
        train_y_tensor = torch.FloatTensor(train_y)
        val_x_tensor = torch.FloatTensor(val_x)
        val_y_tensor = torch.FloatTensor(val_y)
        test_x_tensor = torch.FloatTensor(test_x)
        test_y_tensor = torch.FloatTensor(test_y)
        
        # Create datasets
        train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
        val_dataset = TensorDataset(val_x_tensor, val_y_tensor)
        test_dataset = TensorDataset(test_x_tensor, test_y_tensor)
        
        log.info(f"Created sequence datasets - Train: {len(train_dataset)}, "
                 f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
                 
        return train_dataset, val_dataset, test_dataset


class SequenceDataset(Dataset):
    """
    Dataset for time series sequences.
    """
    def __init__(
        self, 
        data: np.ndarray,
        seq_length: int,
        pred_horizon: int = 1,
        stride: int = 1,
        covariates: Optional[np.ndarray] = None,
        transform=None
    ):
        """
        Initialize sequence dataset.
        
        Args:
            data: Time series data [time_steps, features]
            seq_length: Length of input sequences
            pred_horizon: Number of steps ahead to predict
            stride: Stride between sequences
            covariates: Optional covariate data [time_steps, covariate_features]
            transform: Optional transform to apply to sequences
        """
        self.data = data
        self.seq_length = seq_length
        self.pred_horizon = pred_horizon
        self.stride = stride
        self.covariates = covariates
        self.transform = transform
        
        # Calculate valid indices
        n_samples = data.shape[0]
        self.valid_indices = list(range(0, n_samples - seq_length - pred_horizon + 1, stride))
        
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        # Get start index for the sequence
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.seq_length
        target_idx = end_idx + self.pred_horizon - 1
        
        # Extract sequence
        sequence = self.data[start_idx:end_idx].copy()
        target = self.data[target_idx].copy()
        
        # Include covariates if available
        if self.covariates is not None:
            cov_sequence = self.covariates[start_idx:end_idx].copy()
            sequence = np.concatenate([sequence, cov_sequence], axis=1)
        
        # Apply transforms if any
        if self.transform:
            sequence = self.transform(sequence)
            target = self.transform(target)
            
        return sequence, target
