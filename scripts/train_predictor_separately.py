import os
import logging
import torch
import hydra
from omegaconf import DictConfig, OmegaConf, MISSING # For type hinting and validation
import uuid
from pathlib import Path
import json
from datetime import datetime, timezone # Added for metadata timestamp

# Project imports
from src.utils.setup import setup_logging, setup_seed
from src.data.enhanced_data_loader import EnhancedDataLoader
from src.data.data_processor import DataProcessor
from src.model.autoencoder import AutoencoderModel # For loading AE
from src.model.factory import get_model as get_predictive_model # To get LSTM, Mamba etc.
from src.train.train_model import train_model as train_predictive_model_function # General train function
from src.utils import central_store

log = logging.getLogger(__name__)

def load_autoencoder(cfg: DictConfig, device: torch.device) -> tuple[AutoencoderModel, DictConfig]:
    """Loads a pre-trained Autoencoder model and its configuration."""
    original_cwd = Path(hydra.utils.get_original_cwd())
    central_storage_root_str = cfg.paths.central_model_storage_dir
    if not Path(central_storage_root_str).is_absolute():
        central_storage_root = original_cwd / central_storage_root_str
    else:
        central_storage_root = Path(central_storage_root_str)

    ae_metadata_csv_path = central_storage_root / cfg.paths.ae_metadata_csv_filename
    ae_metadata_csv_path.parent.mkdir(parents=True, exist_ok=True)


    ae_model_path_to_load = cfg.script.get("ae_model_path_to_load", None)
    ae_config_hash_to_load = cfg.script.get("ae_config_hash_to_load", None)

    actual_ae_model_path = None
    ae_config_for_predictor = None

    if ae_model_path_to_load:
        actual_ae_model_path_obj = Path(ae_model_path_to_load)
        if not actual_ae_model_path_obj.is_absolute():
            actual_ae_model_path = original_cwd / actual_ae_model_path_obj
        else:
            actual_ae_model_path = actual_ae_model_path_obj
        log.info(f"Attempting to load AE from direct path: {actual_ae_model_path}")
        ae_config_for_predictor = cfg.model.autoencoder
    elif ae_config_hash_to_load:
        log.info(f"Attempting to load AE with config hash: {ae_config_hash_to_load} from central store.")
        ae_metadata = central_store.find_model_in_metadata_csv(str(ae_metadata_csv_path), ae_config_hash_to_load)
        if not ae_metadata:
            raise ValueError(f"Autoencoder with config hash {ae_config_hash_to_load} not found in metadata at {ae_metadata_csv_path}.")
        actual_ae_model_path = central_storage_root / ae_metadata['model_path_central']
        log.info(f"Found AE in central store. Path: {actual_ae_model_path}")
        try:
            loaded_ae_json_config = json.loads(ae_metadata.get('config_details_json', '{}'))
            # The stored config_details_json for AE was 'relevant_ae_cfg_parts'
            # which had keys like "model_autoencoder", "train_autoencoder"
            if 'model_autoencoder' in loaded_ae_json_config:
                 ae_config_for_predictor = OmegaConf.create(loaded_ae_json_config['model_autoencoder'])
            elif 'latent_dim' in loaded_ae_json_config: # If it was stored flat
                 ae_config_for_predictor = OmegaConf.create(loaded_ae_json_config)
            else:
                 log.warning("Loaded AE config from metadata is not in expected structure. Falling back to current script's AE config.")
                 ae_config_for_predictor = cfg.model.autoencoder
        except json.JSONDecodeError:
            log.warning(f"Failed to parse AE config_details_json from metadata. Using current script's AE config.")
            ae_config_for_predictor = cfg.model.autoencoder
    else:
        raise ValueError("Either 'script.ae_config_hash_to_load' or 'script.ae_model_path_to_load' must be provided.")

    if not actual_ae_model_path or not Path(actual_ae_model_path).exists():
        raise FileNotFoundError(f"AE model file not found at derived path: {actual_ae_model_path}")

    autoencoder = AutoencoderModel(ae_config_for_predictor).to(device)
    autoencoder.load_state_dict(torch.load(actual_ae_model_path, map_location=device))
    autoencoder.eval()
    log.info(f"Successfully loaded pre-trained Autoencoder from: {actual_ae_model_path}")
    return autoencoder, ae_config_for_predictor


@hydra.main(version_base=None, config_path="../conf", config_name="script/config_predictor_separate")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg)
    setup_seed(cfg.experiment.seed if 'experiment' in cfg and 'seed' in cfg.experiment else 42)
    hydra_run_dir = os.getcwd()
    log.info(f"Hydra run output directory for predictor training: {hydra_run_dir}")
    device = torch.device(cfg.experiment.device if 'experiment' in cfg and 'device' in cfg.experiment and torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    writer = None

    active_model_name = cfg.model.name
    if not active_model_name:
        log.error("cfg.model.name must be set (e.g., 'lstm', 'mamba') to specify which predictive model to train.")
        raise ValueError("cfg.model.name not set.")
    log.info(f"Target predictive model: {active_model_name.upper()}")

    loaded_ae_model, loaded_ae_config = load_autoencoder(cfg, device)

    log.info("Loading data and generating latent features using pre-trained AE...")
    data_loader_module = EnhancedDataLoader(cfg)
    raw_data = data_loader_module.load_data()
    data_processor = DataProcessor(cfg)
    processed_data = data_processor.process(raw_data)

    train_dataset_ae, val_dataset_ae, test_dataset_ae = data_processor.create_ae_datasets(processed_data)

    train_latent, val_latent, test_latent = data_processor.generate_latent_representations(
        loaded_ae_model, train_dataset_ae, val_dataset_ae, test_dataset_ae, device
    )
    train_sequence_predictive, val_sequence_predictive, _ = data_processor.create_sequence_datasets(
        train_latent, val_latent, test_latent, processed_data
    )

    predictive_train_cfg = cfg.train[active_model_name]
    train_loader_pred = torch.utils.data.DataLoader(
        train_sequence_predictive, batch_size=predictive_train_cfg.batch_size, shuffle=True,
        num_workers=cfg.data.loader.num_workers, pin_memory=cfg.data.loader.pin_memory
    )
    val_loader_pred = torch.utils.data.DataLoader(
        val_sequence_predictive, batch_size=predictive_train_cfg.batch_size, shuffle=False,
        num_workers=cfg.data.loader.num_workers, pin_memory=cfg.data.loader.pin_memory
    )
    log.info("Latent features generated and data loaders created.")

    ae_ref_for_hash = cfg.script.get("ae_config_hash_to_load") or cfg.script.get("ae_model_path_to_load", "custom_ae_path")

    relevant_predictor_cfg_parts = {
        "model_type": active_model_name,
        "model_network_params": OmegaConf.to_container(cfg.model.network_params, resolve=True),
        "train_predictor_params": OmegaConf.to_container(cfg.train[active_model_name], resolve=True),
        "autoencoder_reference": ae_ref_for_hash,
        "data_sequence_params": OmegaConf.to_container(cfg.data.sequence, resolve=True)
    }
    relevant_predictor_cfg_for_hash = OmegaConf.create(relevant_predictor_cfg_parts)
    predictor_config_hash = central_store.hash_config(relevant_predictor_cfg_for_hash)
    log.info(f"Predictive model ({active_model_name}) configuration hash: {predictor_config_hash}")

    original_cwd = Path(hydra.utils.get_original_cwd())
    central_storage_root_str = cfg.paths.central_model_storage_dir
    if not Path(central_storage_root_str).is_absolute():
        central_storage_root = original_cwd / central_storage_root_str
    else:
        central_storage_root = Path(central_storage_root_str)

    predictor_metadata_csv = central_storage_root / cfg.paths.predictor_metadata_csv_filename
    predictor_metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Predictor Metadata CSV path: {predictor_metadata_csv}")

    existing_predictor_metadata = central_store.find_model_in_metadata_csv(str(predictor_metadata_csv), predictor_config_hash)

    if existing_predictor_metadata and not cfg.script.force_retrain_predictor:
        log.info(f"Existing {active_model_name} model found with hash {predictor_config_hash}.")
        log.info(f"Model ID: {existing_predictor_metadata['model_id']}. Skipping re-training.")
        return

    log.info(f"Initializing and training {active_model_name} model...")

    if not hasattr(loaded_ae_config, 'latent_dim'):
        raise ValueError("Loaded AE configuration does not have 'latent_dim'. Cannot set predictor input/output size.")

    cfg.model.network_params.input_size = loaded_ae_config.latent_dim
    cfg.model.network_params.output_size = loaded_ae_config.latent_dim
    log.info(f"Set {active_model_name} input/output size to AE latent_dim: {loaded_ae_config.latent_dim}")

    predictive_model = get_predictive_model(cfg).to(device)

    training_results = train_predictive_model_function(
        predictive_model, train_loader_pred, val_loader_pred, cfg, device, writer
    )
    log.info(f"{active_model_name} training complete.")
    metrics = training_results if isinstance(training_results, dict) else {}

    log.info(f"Saving {active_model_name} model and updating metadata...")
    experiment_model_save_path = str(Path(hydra_run_dir) / f"trained_{active_model_name}_model.pt")

    central_model_relative_path = central_store.save_model_to__central_store(
        model=predictive_model,
        cfg_hash=predictor_config_hash,
        model_type=active_model_name.upper(),
        central_model_root_dir=str(central_storage_root),
        experiment_model_path=experiment_model_save_path,
        filename=f"{active_model_name}_model.pt"
    )

    metadata = {
        'model_id': str(uuid.uuid4()),
        'config_hash': predictor_config_hash,
        'model_type': active_model_name.upper(),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'model_path_central': central_model_relative_path,
        'model_path_experiment': experiment_model_save_path,
        'config_details_json': OmegaConf.to_json(relevant_predictor_cfg_for_hash, sort_keys=True),
        'metrics_json': json.dumps(metrics),
        'notes': f"Trained via {Path(__file__).name} in run {hydra_run_dir}. AE ref: {ae_ref_for_hash}"
    }
    central_store.update_metadata_csv(str(predictor_metadata_csv), metadata)

    log.info(f"{active_model_name} model saved. Central relative path: {central_model_relative_path}")
    log.info(f"Predictor metadata updated in {predictor_metadata_csv}")
    log.info("Script finished successfully.")

if __name__ == "__main__":
    main()
