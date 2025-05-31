import logging
from omegaconf import DictConfig
import torch.nn as nn

# Import model classes
# Assuming these specific LSTM model classes exist in src.model.lstm
# If not, this part will raise ImportError or a similar issue later.
try:
    from src.model.lstm import LSTMModel, BidirectionalLSTM, AttentionLSTM
except ImportError:
    # Provide dummy classes if real ones are not yet defined to allow file creation
    log_temp = logging.getLogger(__name__ + "_factory_temp") # Corrected typo __name_ to __name__
    log_temp.warning("LSTMModel, BidirectionalLSTM, or AttentionLSTM not found in src.model.lstm. Using placeholders for factory creation.")
    # These placeholders would need to be actual nn.Module subclasses if used.
    class LSTMModel(nn.Module): pass
    class BidirectionalLSTM(nn.Module): pass
    class AttentionLSTM(nn.Module): pass


from src.model.mamba import MambaModel
from src.model.transformer import TransformerModel

log = logging.getLogger(__name__)

def get_model(cfg: DictConfig) -> nn.Module:
    model_name = cfg.model.name.lower() # e.g., 'lstm', 'mamba', 'transformer'

    log.info(f"Attempting to initialize model: {model_name}")

    if model_name == 'lstm':
        # Further check for LSTM variants if your config supports it
        # e.g., cfg.model.lstm.type: 'standard', 'bidirectional', 'attention'
        lstm_type = cfg.model.lstm.get("type", "standard").lower()
        if lstm_type == "standard":
            log.info(f"Initializing Standard LSTM model from factory.")
            return LSTMModel(cfg) # Assumes LSTMModel constructor takes cfg
        elif lstm_type == "bidirectional":
            log.info(f"Initializing Bidirectional LSTM model from factory.")
            return BidirectionalLSTM(cfg) # Assumes BidirectionalLSTM constructor takes cfg
        elif lstm_type == "attention":
            log.info(f"Initializing Attention LSTM model from factory.")
            return AttentionLSTM(cfg) # Assumes AttentionLSTM constructor takes cfg
        else:
            log.error(f"Unknown LSTM type specified in config: {lstm_type}")
            raise ValueError(f"Unknown LSTM type: {lstm_type}")
    elif model_name == 'mamba':
        log.info(f"Initializing Mamba model from factory.")
        return MambaModel(cfg) # Assumes MambaModel constructor takes cfg
    elif model_name == 'transformer':
        log.info(f"Initializing Transformer model from factory.")
        return TransformerModel(cfg) # Assumes TransformerModel constructor takes cfg
    else:
        log.error(f"Unsupported model name specified in config: {model_name}")
        raise ValueError(f"Unsupported model name: {model_name}")

# Optional: Remove get_lstm_model from src/model/lstm.py if it's fully replaced.
# For now, let's keep it to avoid breaking existing code that might use it directly.
