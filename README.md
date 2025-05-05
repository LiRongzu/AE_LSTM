# AE-LSTM Salinity Prediction

This project implements an Autoencoder-LSTM (AE-LSTM) architecture for predicting estuary salinity fields. The model combines dimensionality reduction (via autoencoder) with temporal sequence modeling (via LSTM networks).

## Project Structure

```
AE_LSTM/
├── conf/                 # Configuration files (Hydra)
│   ├── config.yaml       # Main configuration
│   ├── data/             # Data-related configurations
│   ├── model/            # Model architecture configurations
│   ├── paths/            # File paths configurations
│   ├── train/            # Training configurations
│   └── visualization/    # Visualization configurations
├── main_pipeline.py      # Main entry point
├── outputs/              # Generated outputs, logs, models
└── src/                  # Source code
    ├── data/             # Data loading and processing
    ├── generate/         # Data generation utilities
    ├── model/            # Model implementations
    ├── train/            # Training procedures
    ├── utils/            # Utility functions
    └── visualization/    # Visualization functions
```

## Features

- **Modular Design**: Clear separation of data processing, model architecture, and training logic
- **Configurable**: Uses Hydra for configuration management
- **Experiment Tracking**: Supports TensorBoard and Weights & Biases (optional)
- **Hyperparameter Optimization**: Supports Optuna (through configuration)
- **Visualization**: Comprehensive visualization tools for model outputs

## Main Components

### Autoencoder (AE)

The autoencoder module reduces the high-dimensional salinity field data to a lower-dimensional latent space, capturing the essential spatial features of the estuary system.

### LSTM

The Long Short-Term Memory (LSTM) network learns temporal patterns in the latent representations produced by the autoencoder.

### Combined AE-LSTM

The combined model allows for end-to-end training and inference, integrating both spatial and temporal patterns for accurate prediction.

## Usage

### Basic Usage

```bash
python main_pipeline.py
```

### Configuration Overrides

```bash
# Use different model parameters
python main_pipeline.py model=custom_model

# Change training parameters
python main_pipeline.py train.epochs=100 train.batch_size=32

# Change data source
python main_pipeline.py data.dataset.use_mini_dataset=True
```

### Hyperparameter Optimization

```bash
# Run hyperparameter search with Optuna
python main_pipeline.py --multirun train.optimizer.lr=tag(log,interval(1e-4,1e-2)) model.ae.latent_dim=choice(16,32,64)
```

## Requirements

See `requirements.txt` for the complete list of dependencies.

## Development

To extend this project:

1. Add new model architectures in `src/model/`
2. Add configurations in `conf/`
3. Add new dataset types in `src/data/`
4. Add new visualizations in `src/visualization/`
