# AE-LSTM Salinity Prediction

This project implements a multi-architecture neural network framework for predicting estuary salinity fields. The system supports multiple model types including Autoencoder-LSTM (AE-LSTM), Autoencoder-Mamba (AE-Mamba), and Autoencoder-Transformer (AE-Transformer) architectures, combining dimensionality reduction with various temporal sequence modeling approaches.

## Project Structure

```
AE_LSTM/
├── conf/                    # Configuration files (Hydra)
│   ├── config.yaml          # Main configuration
│   ├── data/                # Data-related configurations
│   ├── model/               # Base model configurations
│   ├── model_configs/       # Model-specific configurations
│   │   ├── lstm.yaml        # LSTM architecture parameters
│   │   ├── mamba.yaml       # Mamba architecture parameters
│   │   └── transformer.yaml # Transformer architecture parameters
│   ├── paths/               # File paths configurations
│   ├── train/               # Training configurations
│   └── visualization/       # Visualization configurations
├── main_pipeline.py         # Main entry point
├── outputs/                 # Generated outputs, logs, models
├── requirements.txt         # Project dependencies
└── src/                     # Source code
    ├── data/                # Data loading and processing
    ├── generate/            # Data generation utilities
    ├── model/               # Model implementations
    │   ├── factory.py       # Model factory for dynamic instantiation
    │   ├── lstm.py          # LSTM model variants
    │   ├── mamba.py         # Mamba model implementation
    │   ├── transformer.py   # Transformer model implementation
    │   └── ae_predictive.py # Combined AE-Predictive models
    ├── train/               # Training procedures
    │   └── train_model.py   # Generic training pipeline
    ├── utils/               # Utility functions
    │   └── evaluation.py    # Enhanced evaluation framework
    └── visualization/       # Visualization functions
```

## Features

- **Multi-Architecture Support**: LSTM, Mamba, and Transformer-based temporal models
- **Factory Pattern**: Dynamic model instantiation based on configuration
- **Modular Design**: Clear separation of data processing, model architecture, and training logic
- **Advanced Configuration**: Hydra-based configuration with model-specific parameters
- **Experiment Tracking**: TensorBoard logging and optional Weights & Biases integration
- **Memory Optimization**: Gradient accumulation and mixed precision training support
- **Comprehensive Evaluation**: Advanced metrics and visualization tools
- **Hyperparameter Optimization**: Optuna integration for automated tuning

## Model Architectures

### 1. AE-LSTM
- **Autoencoder**: Reduces high-dimensional salinity fields to latent representations
- **LSTM**: Captures temporal dependencies in the latent space
- **Variants**: Standard LSTM, Bidirectional LSTM, Multi-layer LSTM

### 2. AE-Mamba
- **Autoencoder**: Same dimensionality reduction as LSTM variant
- **Mamba**: State-space model for efficient long-sequence modeling
- **Features**: Linear complexity, selective state updates, hardware-aware design

### 3. AE-Transformer
- **Autoencoder**: Consistent spatial feature extraction
- **Transformer**: Self-attention mechanism for temporal modeling
- **Features**: Multi-head attention, positional encoding, layer normalization

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AE_LSTM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install additional dependencies based on your model choice:
```bash
# For Mamba models
pip install mamba-ssm

# For enhanced optimizations
pip install flash-attn
```

## Usage

### Basic Usage

```bash
# Run with default LSTM configuration
python main_pipeline.py

# Run with specific model type
python main_pipeline.py model_type=mamba
python main_pipeline.py model_type=transformer
```

### Model-Specific Configurations

```bash
# LSTM with specific parameters
python main_pipeline.py model_type=lstm model_configs=lstm train.epochs=100

# Mamba with custom state size
python main_pipeline.py model_type=mamba model_configs=mamba model_configs.d_state=32

# Transformer with different attention heads
python main_pipeline.py model_type=transformer model_configs=transformer model_configs.n_heads=12
```

### Training Configuration

```bash
# Custom training parameters
python main_pipeline.py train.epochs=200 train.batch_size=16 train.learning_rate=0.001

# Enable mixed precision training
python main_pipeline.py train.use_mixed_precision=true

# Enable gradient accumulation
python main_pipeline.py train.gradient_accumulation_steps=4
```

### Data Configuration

```bash
# Use mini dataset for testing
python main_pipeline.py data.dataset.use_mini_dataset=true

# Custom data paths
python main_pipeline.py paths.data_dir=/path/to/data paths.output_dir=/path/to/outputs
```

### Hyperparameter Optimization

```bash
# Multi-run with Optuna
python main_pipeline.py --multirun \
  train.optimizer.lr=tag(log,interval(1e-4,1e-2)) \
  model.ae.latent_dim=choice(16,32,64) \
  model_configs.hidden_size=choice(64,128,256)

# Model comparison
python main_pipeline.py --multirun model_type=lstm,mamba,transformer
```

## Configuration Management

### Main Configuration (`conf/config.yaml`)
- Global settings and default configurations
- Model type selection and general parameters

### Model-Specific Configurations (`conf/model_configs/`)
- **lstm.yaml**: LSTM architecture parameters (hidden_size, num_layers, dropout, bidirectional)
- **mamba.yaml**: Mamba parameters (d_model, d_state, d_conv, expand_factor)
- **transformer.yaml**: Transformer parameters (d_model, n_heads, n_layers, d_ff)

### Training Configuration (`conf/train/default.yaml`)
- Training hyperparameters, optimization settings
- Memory optimization flags, logging preferences

### Path Configuration (`conf/paths/default.yaml`)
- Data directories, output paths, model save locations

## Model Details

### LSTM Architecture
```yaml
model_configs:
  hidden_size: 128        # Hidden state dimensionality
  num_layers: 2           # Number of LSTM layers
  dropout: 0.1            # Dropout probability
  bidirectional: false    # Use bidirectional LSTM
```

### Mamba Architecture
```yaml
model_configs:
  d_model: 128           # Model dimensionality
  d_state: 16            # State space dimensionality
  d_conv: 4              # Convolution kernel size
  expand_factor: 2       # MLP expansion factor
```

### Transformer Architecture
```yaml
model_configs:
  d_model: 128           # Model dimensionality
  n_heads: 8             # Number of attention heads
  n_layers: 6            # Number of transformer layers
  d_ff: 512              # Feed-forward dimensionality
```

## Training Features

- **Automatic Mixed Precision**: Reduces memory usage and speeds up training
- **Gradient Accumulation**: Enables larger effective batch sizes
- **Learning Rate Scheduling**: Cosine annealing and step decay options
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Model Checkpointing**: Automatic saving of best models
- **TensorBoard Logging**: Real-time training monitoring

## Evaluation and Visualization

The framework includes comprehensive evaluation tools:
- **Metrics**: MSE, MAE, R², relative error analysis
- **Visualizations**: Training curves, prediction comparisons, error distributions
- **Model Analysis**: Attention visualization (Transformer), state analysis (Mamba)
- **Performance Profiling**: Memory usage and timing analysis

## Development Guidelines

### Adding New Model Types

1. **Create Model Implementation**: Add new model class in `src/model/`
2. **Update Factory**: Register new model in `src/model/factory.py`
3. **Add Configuration**: Create model-specific config in `conf/model_configs/`
4. **Update Main Config**: Add model type to `conf/config.yaml`

### Example: Adding a New Model
```python
# src/model/new_model.py
from .base_model import BaseModel

class NewModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Implementation details

# src/model/factory.py
def create_model(model_type, config):
    if model_type == "new_model":
        from .new_model import NewModel
        return NewModel(config)
    # ... existing models
```

## Requirements

Key dependencies include:
- PyTorch (≥1.12.0)
- Hydra-core (≥1.3.0)
- TensorBoard
- NumPy, Pandas, Matplotlib
- Optional: mamba-ssm, flash-attn, wandb

See `requirements.txt` for the complete list.

## Recent Updates (v2.0.0)

- **Multi-Model Architecture**: Added support for Mamba and Transformer models
- **Factory Pattern**: Implemented dynamic model instantiation
- **Enhanced Configuration**: Model-specific configuration files
- **Memory Optimization**: Added mixed precision training and gradient accumulation
- **Improved Evaluation**: Advanced metrics and visualization tools
- **Better Logging**: Enhanced TensorBoard integration with model-specific logs
- **Performance Improvements**: Optimized training pipeline for larger models

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add new models or features following the established patterns
4. Update documentation and configuration files
5. Submit a pull request

## License

[Add your license information here]
