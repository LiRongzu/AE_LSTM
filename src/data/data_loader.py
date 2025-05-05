#!/usr/bin/env python
# src/data/data_loader.py - Data loading utilities

import os
import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from omegaconf import DictConfig
import h5py
import xarray as xr
import pandas as pd

log = logging.getLogger(__name__)

class DataLoader:
    """
    Class for loading salinity and related data.
    """
    def __init__(self, cfg: DictConfig):
        """
        Initialize data loader.
        
        Args:
            cfg: Configuration object
        """
        self.cfg = cfg
        self.target_field = cfg.data.dataset.target_field
        self.use_mini_dataset = cfg.data.dataset.use_mini_dataset
        
        # Set data paths
        if self.use_mini_dataset:
            self.data_dir = cfg.paths.mini_data_dir
        else:
            self.data_dir = cfg.paths.raw_data_dir
            
        # Configure mask
        self.apply_mask = cfg.data.preprocessing.apply_mask
        if self.apply_mask:
            self.mask_path = cfg.data.preprocessing.mask_path
        else:
            self.mask_path = None
            
        # Configure covariates
        self.include_covariates = cfg.data.sequence.include_covariates
        self.covariate_fields = cfg.data.sequence.covariate_fields
    
    def load_data(self) -> Dict[str, np.ndarray]:
        """
        Load data from files.
        
        Returns:
            Dictionary containing loaded data arrays
        """
        data = {}
        
        # Load target field (e.g., salinity)
        target_path = os.path.join(self.data_dir, f"{self.target_field}.npy")
        if not os.path.exists(target_path):
            target_path = os.path.join(self.data_dir, f"{self.target_field}_grid.npy")
            
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Target data file not found: {target_path}")
            
        log.info(f"Loading target data from {target_path}")
        data[self.target_field] = np.load(target_path)
        
        # Load mask if specified
        if self.apply_mask and self.mask_path and os.path.exists(self.mask_path):
            log.info(f"Loading mask from {self.mask_path}")
            data['mask'] = np.load(self.mask_path)
        elif self.apply_mask:
            log.warning(f"Mask file not found at {self.mask_path}")
            data['mask'] = None
        
        # Load covariates if specified
        if self.include_covariates:
            for field in self.covariate_fields:
                field_path = os.path.join(self.data_dir, f"{field}.npy")
                if not os.path.exists(field_path):
                    field_path = os.path.join(self.data_dir, f"{field}_grid.npy")
                    
                if os.path.exists(field_path):
                    log.info(f"Loading covariate {field} from {field_path}")
                    data[field] = np.load(field_path)
                else:
                    log.warning(f"Covariate file not found: {field_path}")
        
        # Log data shapes
        for key, arr in data.items():
            log.info(f"Loaded {key} with shape {arr.shape}")
        
        return data
    
    def load_netcdf_data(self, file_path: str) -> Dict[str, np.ndarray]:
        """
        Load data from NetCDF files.
        
        Args:
            file_path: Path to NetCDF file
            
        Returns:
            Dictionary containing loaded data arrays
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"NetCDF file not found: {file_path}")
            
        log.info(f"Loading NetCDF data from {file_path}")
        
        # Open NetCDF file
        ds = xr.open_dataset(file_path)
        
        data = {}
        
        # Extract target field
        if self.target_field in ds.variables:
            data[self.target_field] = ds[self.target_field].values
            
        # Extract covariates
        if self.include_covariates:
            for field in self.covariate_fields:
                if field in ds.variables:
                    data[field] = ds[field].values
        
        # Extract coordinates
        if 'lon' in ds.variables:
            data['lon'] = ds['lon'].values
        if 'lat' in ds.variables:
            data['lat'] = ds['lat'].values
            
        # Close dataset
        ds.close()
        
        # Log data shapes
        for key, arr in data.items():
            log.info(f"Loaded {key} with shape {arr.shape}")
            
        return data
    
    def load_h5_data(self, file_path: str) -> Dict[str, np.ndarray]:
        """
        Load data from HDF5 files.
        
        Args:
            file_path: Path to HDF5 file
            
        Returns:
            Dictionary containing loaded data arrays
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"HDF5 file not found: {file_path}")
            
        log.info(f"Loading HDF5 data from {file_path}")
        
        data = {}
        
        # Open HDF5 file
        with h5py.File(file_path, 'r') as f:
            # Extract target field
            if self.target_field in f:
                data[self.target_field] = f[self.target_field][:]
                
            # Extract covariates
            if self.include_covariates:
                for field in self.covariate_fields:
                    if field in f:
                        data[field] = f[field][:]
        
        # Log data shapes
        for key, arr in data.items():
            log.info(f"Loaded {key} with shape {arr.shape}")
            
        return data

    def load_mask(self) -> Optional[np.ndarray]:
        """
        Load mask array.
        
        Returns:
            Mask array or None if not found
        """
        if not self.apply_mask or not self.mask_path:
            return None
            
        if os.path.exists(self.mask_path):
            log.info(f"Loading mask from {self.mask_path}")
            return np.load(self.mask_path)
        else:
            log.warning(f"Mask file not found: {self.mask_path}")
            return None
