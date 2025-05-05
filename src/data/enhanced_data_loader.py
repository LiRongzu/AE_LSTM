#!/usr/bin/env python
# src/data/enhanced_data_loader.py - Enhanced data loading utilities with improved format support

import os
import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from omegaconf import DictConfig
import h5py
import xarray as xr
import pandas as pd
from scipy.io import loadmat
import zarr
import fsspec
import joblib
from pathlib import Path
import json
from datetime import datetime

log = logging.getLogger(__name__)

class EnhancedDataLoader:
    """
    Enhanced class for loading salinity and related data with extended format support.
    """
    def __init__(self, cfg: DictConfig):
        """
        Initialize enhanced data loader.
        
        Args:
            cfg: Configuration object
        """
        self.cfg = cfg
        self.target_field = cfg.data.dataset.target_field
        self.use_mini_dataset = cfg.data.dataset.get("use_mini_dataset", False)
        
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
        self.covariate_fields = cfg.data.sequence.get("covariate_fields", [])
        
        # Configure remote data access (if applicable)
        self.remote_data = cfg.data.dataset.get("use_remote_data", False)
        if self.remote_data:
            self.remote_protocol = cfg.data.dataset.get("remote_protocol", "s3")
            self.remote_options = cfg.data.dataset.get("remote_options", {})
        
        # Cache settings
        self.use_cache = cfg.data.dataset.get("use_cache", True)
        self.cache_dir = cfg.paths.get("cache_dir", os.path.join(cfg.paths.output_dir, "cache"))
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Track loaded data info
        self.data_info = {}
        
    def load_data(self) -> Dict[str, np.ndarray]:
        """
        Load data from files with smart format detection.
        
        Returns:
            Dictionary containing loaded data arrays
        """
        data = {}
        cache_file = os.path.join(self.cache_dir, f"{self.target_field}_data_cache.npz")
        
        # Try loading from cache first
        if self.use_cache and os.path.exists(cache_file):
            log.info(f"Loading data from cache: {cache_file}")
            cache_data = np.load(cache_file, allow_pickle=True)
            for key in cache_data.files:
                data[key] = cache_data[key]
                log.info(f"Loaded {key} from cache with shape {data[key].shape}")
            return data
            
        # Load target field
        target_files = self._find_field_files(self.target_field)
        if not target_files:
            raise FileNotFoundError(f"Target data file not found for field: {self.target_field}")
            
        target_file = target_files[0]  # Use the first matching file
        log.info(f"Loading target data from {target_file}")
        data[self.target_field] = self._load_file(target_file, self.target_field)
        self.data_info[self.target_field] = {"file": target_file, "shape": data[self.target_field].shape}
        
        # Load mask if specified
        if self.apply_mask and self.mask_path:
            if os.path.exists(self.mask_path):
                log.info(f"Loading mask from {self.mask_path}")
                data['mask'] = self._load_file(self.mask_path, "mask")
                self.data_info['mask'] = {"file": self.mask_path, "shape": data['mask'].shape}
            else:
                log.warning(f"Mask file not found at {self.mask_path}")
                data['mask'] = None
        
        # Load covariates if specified
        if self.include_covariates:
            for field in self.covariate_fields:
                field_files = self._find_field_files(field)
                if field_files:
                    field_file = field_files[0]  # Use the first matching file
                    log.info(f"Loading covariate {field} from {field_file}")
                    data[field] = self._load_file(field_file, field)
                    self.data_info[field] = {"file": field_file, "shape": data[field].shape}
                else:
                    log.warning(f"Covariate file not found for field: {field}")
        
        # Log data shapes
        for key, arr in data.items():
            log.info(f"Loaded {key} with shape {arr.shape}")
        
        # Save to cache if enabled
        if self.use_cache:
            log.info(f"Saving loaded data to cache: {cache_file}")
            np.savez_compressed(cache_file, **data)
            
            # Also save data info as JSON
            with open(os.path.join(self.cache_dir, "data_info.json"), "w") as f:
                info_dict = {k: {"shape": list(v["shape"]), "file": v["file"]} 
                           for k, v in self.data_info.items()}
                info_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                json.dump(info_dict, f, indent=2)
        
        return data
    
    def _find_field_files(self, field: str) -> List[str]:
        """
        Find files matching a field name with various possible extensions.
        
        Args:
            field: Field name to search for
            
        Returns:
            List of matching file paths
        """
        patterns = [
            f"{field}.npy", 
            f"{field}_grid.npy", 
            f"{field}.nc", 
            f"{field}.h5", 
            f"{field}.h5", 
            f"{field}.mat", 
            f"{field}.zarr",
            f"{field}.csv",
            f"{field}.parquet"
        ]
        
        found_files = []
        
        # If remote data is enabled, use fsspec
        if self.remote_data:
            fs = fsspec.filesystem(self.remote_protocol, **self.remote_options)
            for pattern in patterns:
                remote_path = os.path.join(self.data_dir, pattern)
                if fs.exists(remote_path):
                    found_files.append(remote_path)
        else:
            # Search in local filesystem
            for pattern in patterns:
                local_path = os.path.join(self.data_dir, pattern)
                if os.path.exists(local_path):
                    found_files.append(local_path)
                    
        return found_files
    
    def _load_file(self, file_path: str, field_name: str = None) -> np.ndarray:
        """
        Load data from file with automatic format detection.
        
        Args:
            file_path: Path to data file
            field_name: Optional field name for multi-field files
            
        Returns:
            Numpy array of loaded data
        """
        # Handle remote files
        if self.remote_data and not os.path.exists(file_path):
            fs = fsspec.filesystem(self.remote_protocol, **self.remote_options)
            with fs.open(file_path, 'rb') as f:
                return self._load_from_buffer(f, file_path, field_name)
        
        # Handle local files based on extension
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.npy':
            return np.load(file_path)
        elif ext == '.npz':
            data = np.load(file_path)
            if field_name and field_name in data.files:
                return data[field_name]
            else:
                # Return the first array if no field name specified
                return data[list(data.files)[0]]
        elif ext == '.nc':
            return self._load_netcdf(file_path, field_name)
        elif ext == '.h5' or ext == '.hdf5':
            return self._load_hdf5(file_path, field_name)
        elif ext == '.mat':
            mat_data = loadmat(file_path)
            if field_name and field_name in mat_data:
                return mat_data[field_name]
            else:
                # Find the first array that doesn't start with '__' (metadata)
                for key in mat_data:
                    if not key.startswith('__'):
                        return mat_data[key]
        elif ext == '.zarr':
            z = zarr.open(file_path, mode='r')
            if field_name and field_name in z:
                return z[field_name][:]
            else:
                # Return the first array
                for key in z:
                    return z[key][:]
        elif ext == '.csv':
            df = pd.read_csv(file_path)
            if field_name and field_name in df.columns:
                return df[field_name].values
            else:
                return df.values
        elif ext == '.parquet':
            df = pd.read_parquet(file_path)
            if field_name and field_name in df.columns:
                return df[field_name].values
            else:
                return df.values
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    
    def _load_from_buffer(self, buffer, file_path: str, field_name: str = None) -> np.ndarray:
        """Load data from an open file buffer."""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.npy':
            return np.load(buffer)
        elif ext == '.nc':
            ds = xr.open_dataset(buffer)
            data = ds[field_name].values if field_name and field_name in ds else None
            ds.close()
            return data
        # Add more format handlers as needed
        else:
            raise ValueError(f"Unsupported file extension for buffer loading: {ext}")
    
    def _load_netcdf(self, file_path: str, field_name: str = None) -> np.ndarray:
        """Load data from NetCDF file."""
        ds = xr.open_dataset(file_path)
        
        if field_name and field_name in ds.variables:
            data = ds[field_name].values
        else:
            # Get the first non-dimension variable
            for var_name in ds.variables:
                if var_name not in ds.dims:
                    data = ds[var_name].values
                    break
            else:
                raise ValueError(f"No valid data variables found in {file_path}")
                
        ds.close()
        return data
    
    def _load_hdf5(self, file_path: str, field_name: str = None) -> np.ndarray:
        """Load data from HDF5 file."""
        with h5py.File(file_path, 'r') as f:
            if field_name and field_name in f:
                return f[field_name][:]
            else:
                # Get the first dataset
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        return f[key][:]
                raise ValueError(f"No dataset found in {file_path}")

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about loaded data.
        
        Returns:
            Dictionary of metadata
        """
        metadata = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_dir": self.data_dir,
            "fields": self.data_info,
        }
        
        return metadata
