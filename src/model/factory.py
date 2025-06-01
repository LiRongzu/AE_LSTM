import logging
from omegaconf import DictConfig, OmegaConf
import torch.nn as nn

# Import model classes
try:
    from src.model.lstm import LSTMModel, BidirectionalLSTM, AttentionLSTM
except ImportError:
    log_temp = logging.getLogger(__name__ + "_factory_temp")
    log_temp.warning("LSTMModel, BidirectionalLSTM, or AttentionLSTM not found in src.model.lstm. Using placeholders.")
    class LSTMModel(nn.Module):
        def __init__(self, model_cfg: DictConfig):
            super().__init__()
            log_temp.info(f"Placeholder LSTMModel initialized")
    class BidirectionalLSTM(nn.Module):
        def __init__(self, model_cfg: DictConfig):
            super().__init__()
            log_temp.info(f"Placeholder BidirectionalLSTM initialized")
    class AttentionLSTM(nn.Module):
        def __init__(self, model_cfg: DictConfig):
            super().__init__()
            log_temp.info(f"Placeholder AttentionLSTM initialized")

from src.model.mamba import MambaModel
from src.model.transformer import TransformerModel

log = logging.getLogger(__name__)

def get_model(cfg: DictConfig) -> nn.Module:
    model_name = cfg.model.name.lower() # e.g., 'lstm', 'mamba', 'transformer'
    # cfg.model now directly contains the specific parameters for the active model
    # (e.g., input_size, hidden_size for lstm, d_model for mamba, etc.)
    # as they are merged from conf/model/model_configs/<model_name>.yaml into cfg.model
    model_specific_cfg = cfg.model

    log.info(f"Attempting to initialize model: {model_name}")
    log.debug(f"Model configuration: {OmegaConf.to_yaml(model_specific_cfg)}")

    if model_name == 'lstm':
        # LSTM type is now directly in cfg.model.type (not cfg.model.lstm.type)
        lstm_type = model_specific_cfg.get("type", "standard").lower()
        
        if lstm_type == "standard":
            log.info(f"Initializing Standard LSTM model from factory.")
            return LSTMModel(model_specific_cfg)
        elif lstm_type == "bidirectional":
            log.info(f"Initializing Bidirectional LSTM model from factory.")
            return BidirectionalLSTM(model_specific_cfg)
        elif lstm_type == "attention":
            log.info(f"Initializing Attention LSTM model from factory.")
            return AttentionLSTM(model_specific_cfg)
        else:
            log.error(f"Unknown LSTM type specified in config: {lstm_type}")
            raise ValueError(f"Unknown LSTM type: {lstm_type}")
            
    elif model_name == 'mamba':
        log.info(f"Initializing Mamba model from factory.")
        return MambaModel(model_specific_cfg)
        
    elif model_name == 'transformer':
        log.info(f"Initializing Transformer model from factory.")
        return TransformerModel(model_specific_cfg)
        
    else:
        log.error(f"Unsupported model name specified in config: {model_name}")
        raise ValueError(f"Unsupported model name: {model_name}")

# Optional: Remove get_lstm_model from src/model/lstm.py if it's fully replaced.
# For now, let's keep it to avoid breaking existing code that might use it directly.
