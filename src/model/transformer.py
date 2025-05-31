import torch
import torch.nn as nn
import math
from omegaconf import DictConfig, OmegaConf # Added OmegaConf for the test case
import logging

# Configure logger
# Basic configuration for the logger (can be more sophisticated in a real app)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # (max_len, 1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim] if batch_first=True for TransformerEncoder
               or shape [seq_len, batch_size, embedding_dim] if batch_first=False
        """
        # Assuming x is [batch_size, seq_len, embedding_dim] (batch_first=True)
        # We need to add pe (max_len, 1, d_model) to x (batch, seq, d_model)
        # pe.squeeze(1) gives (max_len, d_model). We need (seq_len, d_model) then unsqueeze for batch.
        # Or, directly use pe[:x.size(1), :] which is (seq_len, 1, d_model) and let it broadcast.
        x = x + self.pe[:x.size(1), :].transpose(0,1) # x is (B,S,D), pe[:S,:] is (S,1,D) -> (1,S,D) for broadcasting
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(TransformerModel, self).__init__()
        self.cfg = cfg
        self.model_cfg = cfg.model.transformer

        self.input_size = self.model_cfg.input_size
        self.d_model = self.model_cfg.d_model
        self.nhead = self.model_cfg.nhead
        self.num_encoder_layers = self.model_cfg.num_encoder_layers
        self.dim_feedforward = self.model_cfg.dim_feedforward
        self.dropout = self.model_cfg.dropout
        self.output_size = self.model_cfg.output_size

        self.input_fc = nn.Linear(self.input_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout)

        # TransformerEncoderLayer expects d_model, nhead, dim_feedforward, dropout
        # batch_first=True means input/output tensors are (batch, seq, feature)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=self.num_encoder_layers
        )
        self.output_fc = nn.Linear(self.d_model, self.output_size)

        self._init_weights()
        log.info(f"TransformerModel initialized with: input_size={self.input_size}, d_model={self.d_model}, nhead={self.nhead}, "
                   f"num_encoder_layers={self.num_encoder_layers}, dim_feedforward={self.dim_feedforward}, "
                   f"dropout={self.dropout}, output_size={self.output_size}")

    def _init_weights(self):
        initrange = 0.1
        self.input_fc.weight.data.uniform_(-initrange, initrange)
        if self.input_fc.bias is not None:
            self.input_fc.bias.data.zero_()

        self.output_fc.weight.data.uniform_(-initrange, initrange)
        if self.output_fc.bias is not None:
            self.output_fc.bias.data.zero_()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src shape: (batch_size, seq_len, input_size)
        src = self.input_fc(src) * math.sqrt(self.d_model) # Project to d_model and scale
        src = self.pos_encoder(src) # Add positional encoding

        # src is (batch_size, seq_len, d_model) due to batch_first=True
        output = self.transformer_encoder(src) # (batch_size, seq_len, d_model)

        # We need to predict the next latent state, so we take the output of the last time step
        output = output[:, -1, :] # (batch_size, d_model)
        output = self.output_fc(output) # (batch_size, output_size)
        return output

if __name__ == '__main__':
    # Create a dummy Hydra DictConfig for model.transformer
    dummy_cfg = OmegaConf.create({
        "model": {
            "transformer": {
                "input_size": 32,       # Example: latent_dim
                "d_model": 64,          # Dimension of the transformer model
                "nhead": 4,             # Number of attention heads
                "num_encoder_layers": 3,# Number of encoder layers
                "dim_feedforward": 128, # Dimension of feedforward network
                "dropout": 0.1,
                "output_size": 32       # Example: latent_dim (should match input_size for this use case)
            }
        }
    })

    log.info("Starting TransformerModel test...")
    # Instantiate TransformerModel
    try:
        model = TransformerModel(dummy_cfg)
        log.info("TransformerModel instantiated successfully.")
    except Exception as e:
        log.error(f"Error instantiating TransformerModel: {e}", exc_info=True)
        exit()

    # Create a dummy input tensor
    batch_size = 4
    seq_len = 10 # Example sequence length
    input_size = dummy_cfg.model.transformer.input_size

    dummy_input = torch.randn(batch_size, seq_len, input_size)
    log.info(f"Dummy input shape: {dummy_input.shape}")

    # Pass it through the model
    try:
        output = model(dummy_input)
        log.info(f"Output shape: {output.shape}")

        # Expected output shape (batch_size, output_size) because we take the last time step
        expected_output_shape = (batch_size, dummy_cfg.model.transformer.output_size)
        if output.shape == expected_output_shape:
            log.info("TransformerModel test successful! Output shape matches expected shape.")
        else:
            log.error(f"TransformerModel test FAILED. Output shape: {output.shape}, Expected: {expected_output_shape}")

    except Exception as e:
        log.error(f"Error during model forward pass: {e}", exc_info=True)

    log.info("TransformerModel test finished.")
