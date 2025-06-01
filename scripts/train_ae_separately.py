import os
import logging
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import uuid # For generating unique model IDs
from pathlib import Path
from datetime import datetime, timezone
import json

# Project imports
from src.utils.setup import setup_logging, setup_seed # setup_experiment removed for now
from src.data.enhanced_data_loader import EnhancedDataLoader
from src.data.data_processor import DataProcessor
from src.model.autoencoder import AutoencoderModel
from src.train.train_autoencoder import train_autoencoder
from src.utils import central_store # Import the new utilities

# Initialize logger for this script
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="script/config_ae_separate")
def main(cfg: DictConfig) -> None:
    """
    Standalone script to train an Autoencoder, manage it with a central store,
    and avoid retraining if an identical model exists.
    """
    # 1. Initial Setup
    setup_logging(cfg)
    setup_seed(cfg.experiment.seed if 'experiment' in cfg and 'seed' in cfg.experiment else 42)

    hydra_run_dir = os.getcwd()
    log.info(f"Hydra run output directory: {hydra_run_dir}")

    device = torch.device(cfg.experiment.device if 'experiment' in cfg and 'device' in cfg.experiment and torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    writer = None # Placeholder for Tensorboard SummaryWriter

    # 2. Prepare Configuration for Hashing
    relevant_ae_cfg_parts = {
        "model_autoencoder": cfg.model.autoencoder,
        "train_autoencoder": cfg.train.autoencoder,
        "data_sequence": cfg.data.sequence
    }
    relevant_ae_cfg_for_hash = OmegaConf.create(relevant_ae_cfg_parts)
    config_hash = central_store.hash_config(relevant_ae_cfg_for_hash)
    log.info(f"Autoencoder configuration hash: {config_hash}")

    # 3. Central Store Check
    original_cwd = Path(hydra.utils.get_original_cwd())
    # Ensure central_model_storage_dir is treated as relative to original_cwd if not absolute
    central_storage_path_obj = Path(cfg.paths.central_model_storage_dir)
    if not central_storage_path_obj.is_absolute():
        central_storage_path_obj = original_cwd / central_storage_path_obj

    metadata_csv_path = central_storage_path_obj / cfg.paths.ae_metadata_csv_filename

    log.info(f"AE Metadata CSV path: {metadata_csv_path}")
    metadata_csv_path.parent.mkdir(parents=True, exist_ok=True) # Ensure metadata dir exists

    existing_model_metadata = central_store.find_model_in_metadata_csv(str(metadata_csv_path), config_hash)

    if existing_model_metadata and not cfg.script.force_retrain:
        log.info(f"Existing Autoencoder model found with matching config hash: {config_hash}")
        log.info(f"Model ID: {existing_model_metadata['model_id']}")
        log.info(f"Central Path: {existing_model_metadata['model_path_central']}")
        log.info("Skipping re-training.")
        return

    # 4. Load Data
    log.info("Loading and processing data for Autoencoder training...")
    data_loader = EnhancedDataLoader(cfg)
    raw_data = data_loader.load_data()
    data_processor = DataProcessor(cfg)
    processed_data = data_processor.process(raw_data)
    train_dataset_ae, val_dataset_ae, _ = data_processor.create_ae_datasets(processed_data)

    train_loader_ae = torch.utils.data.DataLoader(
        train_dataset_ae, batch_size=cfg.train.autoencoder.batch_size, shuffle=True,
        num_workers=cfg.data.loader.num_workers, pin_memory=cfg.data.loader.pin_memory
    )
    val_loader_ae = torch.utils.data.DataLoader(
        val_dataset_ae, batch_size=cfg.train.autoencoder.batch_size, shuffle=False,
        num_workers=cfg.data.loader.num_workers, pin_memory=cfg.data.loader.pin_memory
    )
    log.info("Data loading complete.")

    # 5. Model Training
    log.info("Initializing and training Autoencoder model...")
    autoencoder = AutoencoderModel(cfg.model.autoencoder).to(device) # Pass only relevant AE config

    training_results = train_autoencoder(autoencoder, train_loader_ae, val_loader_ae, cfg, device, writer)
    log.info("Autoencoder training complete.")
    metrics = training_results if isinstance(training_results, dict) else {}

    # 6. Model Saving & Metadata Update
    log.info("Saving model and updating metadata...")
    experiment_model_save_path = str(Path(hydra_run_dir) / "trained_ae_model.pt")

    # Ensure central_storage_path_obj is used for saving
    central_model_relative_path = central_store.save_model_to_central_store(
        model=autoencoder,
        cfg_hash=config_hash,
        model_type="Autoencoder",
        central_model_root_dir=str(central_storage_path_obj),
        experiment_model_path=experiment_model_save_path,
        filename="ae_model.pt"
    )

    metadata = {
        'model_id': str(uuid.uuid4()),
        'config_hash': config_hash,
        'model_type': "Autoencoder",
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'model_path_central': central_model_relative_path,
        'model_path_experiment': experiment_model_save_path,
        'config_details_json': OmegaConf.to_json(relevant_ae_cfg_for_hash, sort_keys=True),
        'metrics_json': json.dumps(metrics),
        'notes': f"Trained via {Path(__file__).name} in run: {hydra_run_dir}"
    }
    central_store.update_metadata_csv(str(metadata_csv_path), metadata)

    log.info(f"Autoencoder saved. Central relative path: {central_model_relative_path}")
    log.info(f"Metadata updated in {metadata_csv_path}")
    log.info("Script finished successfully.")

if __name__ == "__main__":
    main()
