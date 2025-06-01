import os
import hashlib
import json
import csv
from pathlib import Path
from datetime import datetime, timezone
from omegaconf import DictConfig, OmegaConf
import torch
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# --- Configuration Hashing ---
def _config_to_serializable_dict(cfg: DictConfig) -> Dict[str, Any]:
    """
    Converts relevant parts of an OmegaConf DictConfig to a sorted, serializable dictionary.
    This ensures consistent hashing.
    We should only include parameters that define the model's architecture and training uniqueness.
    Excludes paths, run-specific dirs, etc.
    """
    # Example: Select specific sub-configs that define the model.
    # This needs to be tailored based on what defines model uniqueness.
    # For an AE, it might be cfg.model.autoencoder and relevant parts of cfg.train.autoencoder.
    # For a predictor, it might be cfg.model.network_params and relevant parts of cfg.train[model_name].

    # For now, let's assume the passed cfg IS the relevant part for hashing.
    # If cfg contains dynamic elements like absolute paths or timestamps not relevant to model architecture,
    # they should be excluded or normalized here.

    # Convert OmegaConf to a primitive Python dict
    primitive_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)

    # Simple approach: serialize the whole primitive_dict.
    # More robust: cherry-pick specific keys that define the model.
    # For this initial version, we'll serialize what's passed, assuming it's pre-filtered.
    return primitive_dict

def hash_config(relevant_cfg: DictConfig) -> str:
    """
    Generates a SHA256 hash for a given configuration.
    'relevant_cfg' should be a DictConfig containing only the parameters
    that define the model's uniqueness.
    """
    serializable_dict = _config_to_serializable_dict(relevant_cfg)
    # Sort the dictionary by keys to ensure consistent byte string for hashing
    # and use compact JSON representation.
    config_str = json.dumps(serializable_dict, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(config_str.encode('utf-8')).hexdigest()

# --- Model Saving ---
def save_model_to_central_store(
    model: torch.nn.Module,
    cfg_hash: str,
    model_type: str, # e.g., "Autoencoder", "LSTM"
    central_model_root_dir: str, # From cfg.paths.central_model_storage_dir
    experiment_model_path: str, # Full path where model is also saved for the experiment
    filename: str = "model.pt"
) -> str:
    """
    Saves the model to the central store and ensures it's saved to the experiment path.
    Returns the relative path of the model in the central store.
    """
    # Ensure experiment model is saved
    experiment_model_path_obj = Path(experiment_model_path)
    experiment_model_path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), experiment_model_path_obj)
    logger.info(f"Model saved to experiment path: {experiment_model_path_obj}")

    # Prepare central storage path
    # Example: <central_root_dir>/Autoencoder/<cfg_hash>/model.pt
    central_model_dir = Path(central_model_root_dir) / model_type / cfg_hash
    central_model_dir.mkdir(parents=True, exist_ok=True)
    central_model_file_path = central_model_dir / filename

    torch.save(model.state_dict(), central_model_file_path)
    logger.info(f"Model saved to central store: {central_model_file_path}")

    # Return the path relative to the central_model_root_dir for storage in metadata
    relative_central_path = str(central_model_file_path.relative_to(Path(central_model_root_dir)))
    return relative_central_path

# --- Metadata Management (CSV) ---
CSV_FIELDNAMES = [
    'model_id', 'config_hash', 'model_type', 'timestamp_utc',
    'model_path_central', 'model_path_experiment',
    'config_details_json', 'metrics_json', 'notes'
]

def update_metadata_csv(
    csv_full_path: str, # Full path to the CSV file
    metadata_record: Dict[str, Any]
) -> None:
    """
    Appends a new record to the CSV file.
    Creates the CSV and writes headers if it doesn't exist.
    """
    csv_path_obj = Path(csv_full_path)
    csv_path_obj.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path_obj.is_file()

    # Ensure all fields are present in the record, defaulting to None or empty string
    for fieldname in CSV_FIELDNAMES:
        metadata_record.setdefault(fieldname, "")

    try:
        with open(csv_path_obj, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            writer.writerow(metadata_record)
        logger.info(f"Metadata successfully updated in: {csv_full_path}")
    except IOError as e:
        logger.error(f"Failed to update metadata CSV {csv_full_path}: {e}")
        raise

def find_model_in_metadata_csv(
    csv_full_path: str, # Full path to the CSV file
    config_hash: str
) -> Optional[Dict[str, str]]:
    """
    Searches the CSV for a model with the given config_hash.
    Returns the first matching record (dictionary) or None if not found.
    """
    csv_path_obj = Path(csv_full_path)
    if not csv_path_obj.is_file():
        logger.warning(f"Metadata CSV file not found: {csv_full_path}")
        return None

    try:
        with open(csv_path_obj, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['config_hash'] == config_hash:
                    logger.info(f"Found model with config_hash {config_hash} in {csv_full_path}")
                    return row
        logger.info(f"No model found with config_hash {config_hash} in {csv_full_path}")
        return None
    except IOError as e:
        logger.error(f"Failed to read metadata CSV {csv_full_path}: {e}")
        raise
    except Exception as e: # Catch other potential errors like empty CSV or missing headers after creation
        logger.error(f"Error processing CSV file {csv_full_path}: {e}")
        return None


if __name__ == '__main__':
    # Example Usage (for testing the module directly)
    print("Testing central_store.py utilities...")

    # Dummy config
    dummy_hydra_cfg = OmegaConf.create({
        "model": {
            "type": "dummy",
            "layers": [64, 32],
            "lr": 0.001
        },
        "data": {
            "path": "/data/dummy" # This path might change, should ideally be excluded from hash if not relevant
        }
    })

    # For hashing, you'd typically select the relevant part of the config
    # For example, if only model structure matters for uniqueness:
    relevant_model_cfg = dummy_hydra_cfg.model

    cfg_hash = hash_config(relevant_model_cfg)
    print(f"Generated Config Hash: {cfg_hash}")

    # Dummy metadata paths (replace with actual config values in real use)
    TEST_OUTPUT_DIR = Path("./test_outputs")
    CENTRAL_STORAGE_ROOT = TEST_OUTPUT_DIR / "central_storage"
    METADATA_DIR = TEST_OUTPUT_DIR / "metadata"
    AE_METADATA_CSV = METADATA_DIR / "ae_test_metadata.csv"

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    CENTRAL_STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

    # Test find_model (on a potentially non-existent or empty file)
    print(f"Searching for model before adding: {find_model_in_metadata_csv(str(AE_METADATA_CSV), cfg_hash)}")

    # Dummy model
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10,1)
        def forward(self, x):
            return self.linear(x)

    dummy_torch_model = DummyModel()

    # Test save_model
    experiment_save_path = str(TEST_OUTPUT_DIR / "experiments" / "run123" / "ae_model.pt")

    # Ensure the experiment save directory exists
    Path(experiment_save_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        central_path = save_model_to_central_store(
            model=dummy_torch_model,
            cfg_hash=cfg_hash,
            model_type="Autoencoder",
            central_model_root_dir=str(CENTRAL_STORAGE_ROOT),
            experiment_model_path=experiment_save_path
        )
        print(f"Model saved to central path: {central_path}")

        # Test update_metadata
        import uuid
        metadata = {
            'model_id': str(uuid.uuid4()),
            'config_hash': cfg_hash,
            'model_type': "Autoencoder",
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'model_path_central': central_path,
            'model_path_experiment': experiment_save_path,
            'config_details_json': json.dumps(OmegaConf.to_container(relevant_model_cfg, resolve=True)),
            'metrics_json': json.dumps({"loss": 0.1234, "accuracy": 0.95}),
            'notes': "Test run for AE model"
        }
        update_metadata_csv(str(AE_METADATA_CSV), metadata)

        # Test find_model again
        found_record = find_model_in_metadata_csv(str(AE_METADATA_CSV), cfg_hash)
        print(f"Searching for model after adding: {found_record}")
        if found_record:
            assert found_record['config_hash'] == cfg_hash
            assert found_record['model_path_central'] == central_path

        print(f"Metadata CSV created at: {AE_METADATA_CSV}")
        print(f"Central model saved under: {CENTRAL_STORAGE_ROOT / central_path}")
        print("Test completed. Check test_outputs directory.")

    except Exception as e:
        print(f"An error occurred during testing: {e}")
        import traceback
        traceback.print_exc()
