import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
import logging
from dataclasses import dataclass
from einops import rearrange, repeat, einsum # This will require einops to be installed

logger = logging.getLogger(__name__)

# --- Helper Dataclass (from johnma2006/mamba-minimal) ---
@dataclass
class MinimalMambaArgs:
    d_model: int
    n_layer: int
    # vocab_size: int # Removed, as we're dealing with continuous features, not tokens
    d_state: int = 16
    expand: int = 2
    dt_rank: str | int = 'auto' # Union type for Python 3.9+
    d_conv: int = 4
    # pad_vocab_size_multiple: int = 8 # Removed
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
        # vocab_size adjustment removed

# --- RMSNorm (from johnma2006/mamba-minimal) ---
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output

# --- MambaBlock (from johnma2006/mamba-minimal, adapted for no vocab/embedding) ---
class MambaBlock(nn.Module):
    def __init__(self, args: MinimalMambaArgs):
        super().__init__()
        self.args = args
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        (b, l, d) = x.shape
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        x = F.silu(x)
        y = self.ssm(x)
        y = y * F.silu(res)
        output = self.out_proj(y)
        return output

    def ssm(self, x):
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        x_dbl = self.x_proj(x)
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        y = self.selective_scan(x, delta, A, B, C, D)
        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        x_state = torch.zeros((b, d_in, n), device=deltaA.device) # Changed name from x to x_state to avoid confusion
        ys = []
        for i in range(l):
            x_state = deltaA[:, i] * x_state + deltaB_u[:, i]
            y_i = einsum(x_state, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y_i)
        y = torch.stack(ys, dim=1)
        y = y + u * D
        return y

# --- ResidualBlock (from johnma2006/mamba-minimal) ---
class ResidualBlock(nn.Module):
    def __init__(self, args: MinimalMambaArgs):
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        output = self.mixer(self.norm(x)) + x
        return output

# --- Main MambaModel (adapted from johnma2006/mamba-minimal Mamba class) ---
class MambaModel(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()

        # Extract parameters from model config (now cfg.model directly)
        args = MinimalMambaArgs(
            d_model=model_cfg.d_model,
            n_layer=model_cfg.n_layer,
            d_state=model_cfg.d_state,
            expand=model_cfg.expand,
            dt_rank=model_cfg.dt_rank if hasattr(model_cfg, 'dt_rank') else 'auto',
            d_conv=model_cfg.d_conv,
            conv_bias=model_cfg.conv_bias if hasattr(model_cfg, 'conv_bias') else True,
            bias=model_cfg.bias if hasattr(model_cfg, 'bias') else False,
        )
        self.args = args

        # Removed embedding layer: self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        # Input x will be (batch_size, seq_len, d_model) directly

        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)

        # Removed lm_head as this model is for feature extraction/prediction, not language modeling
        # if model_cfg.tie_weights and hasattr(self.embedding, 'weight'):
        #    self.lm_head.weight = self.embedding.weight

        # Add the output linear layer
        self.output_size = model_cfg.output_size # Get output_size from config
        self.out_fc = nn.Linear(self.args.d_model, self.output_size)

        logger.info(f"MambaModel (minimal) initialized with args: {self.args}, output_size: {self.output_size}")

    def forward(self, x):
        # Input x is expected to be (batch_size, seq_len, d_model)
        # No embedding needed if input is already features.

        # logger.debug(f"MambaModel input shape: {x.shape}")

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x) # Shape (batch_size, seq_len, d_model)

        # Take the output of the last time step
        x_last_step = x[:, -1, :] # Shape (batch_size, d_model)

        # Apply final linear layer
        output = self.out_fc(x_last_step) # Shape (batch_size, output_size)

        # logger.debug(f"MambaModel output shape: {output.shape}")
        return output

# --- Example Usage ---
if __name__ == '__main__':
    # Create a dummy Hydra DictConfig
    # Ensure these match what MambaModel expects from cfg.model.mamba
    # Add output_size to the dummy config for the __main__ test
    cfg = OmegaConf.create({
        "model": {
            "mamba": {
                "d_model": 64,    # Corresponds to input_size for time series
                "n_layer": 2,
                "output_size": 32, # Example output size
                "d_state": 16,    # Standard Mamba param
                "expand": 2,      # Standard Mamba param
                "d_conv": 4,      # Standard Mamba param
                # dt_rank, conv_bias, bias can be omitted to use defaults in MinimalMambaArgs
            }
        },
        # "tie_weights": False # Example if we had lm_head and wanted to control tying
    })

    # Instantiate the model
    model = MambaModel(cfg.model)

    # Create a dummy input tensor
    batch_size = 4
    seq_len = 20
    input_size = cfg.model.d_model # d_model is the feature size per timestep

    dummy_input = torch.randn(batch_size, seq_len, input_size)
    logger.info(f"Test input shape: {dummy_input.shape}")

    # Forward pass
    try:
        output = model(dummy_input)
        logger.info(f"Test output shape: {output.shape}")

        # Check if output shape is as expected (batch_size, output_size)
        expected_output_shape = (batch_size, cfg.model.output_size)
        if output.shape == expected_output_shape:
            logger.info("MambaModel (minimal) with output layer test successful!")
        else:
            logger.error(f"MambaModel (minimal) with output layer test FAILED. Output shape: {output.shape}, Expected: {expected_output_shape}")

    except ImportError as e:
        logger.error(f"Test failed due to ImportError: {e}. Make sure 'einops' is installed.")
    except Exception as e:
        logger.error(f"An error occurred during the test: {e}", exc_info=True)
