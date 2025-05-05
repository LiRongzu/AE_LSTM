# AE-LSTM 盐度预测

本项目实现了一个自编码器-长短期记忆网络 (AE-LSTM) 架构，用于预测河口盐度场。该模型结合了降维（通过自编码器）和时间序列建模（通过 LSTM 网络）。

## 项目结构

```
AE_LSTM/
├── conf/                 # 配置文件 (Hydra)
│   ├── config.yaml       # 主配置文件
│   ├── data/             # 数据相关配置
│   ├── model/            # 模型架构配置
│   ├── paths/            # 文件路径配置
│   ├── train/            # 训练配置
│   └── visualization/    # 可视化配置
├── main_pipeline.py      # 主程序入口
├── outputs/              # 生成的输出、日志、模型等
└── src/                  # 源代码
    ├── data/             # 数据加载与处理
    ├── generate/         # 数据生成工具
    ├── model/            # 模型实现
    ├── train/            # 训练流程
    ├── utils/            # 工具函数
    └── visualization/    # 可视化函数
```

## 特性

- **模块化设计**: 清晰分离数据处理、模型架构和训练逻辑
- **可配置**: 使用 Hydra 进行配置管理
- **实验跟踪**: 支持 TensorBoard 和 Weights & Biases (可选)
- **超参数优化**: 支持 Optuna (通过配置)
- **可视化**: 提供全面的模型输出可视化工具

## 主要组件

### 自编码器 (Autoencoder - AE)

自编码器模块将高维的盐度场数据降维到低维的潜在空间，捕捉河口系统的关键空间特征。

### 长短期记忆网络 (LSTM)

长短期记忆 (LSTM) 网络学习由自编码器产生的潜在表示中的时间模式。

### 组合 AE-LSTM 模型

组合模型允许进行端到端的训练和推理，整合空间和时间模式以实现准确预测。

## 详细流程说明

本项目的核心流程 (`main_pipeline.py`) 通过 Hydra 配置驱动，完成从数据加载到模型评估的完整步骤。以下是详细说明：

1.  **环境与配置初始化 (`setup_experiment`, `@hydra.main`)**
    *   **目的**: 设置实验环境，加载配置。
    *   **操作**: Hydra 解析 `conf/` 目录下的 YAML 文件，特别是 `config.yaml` 及其引用的 `defaults`。
    *   **关键参数**: `experiment.seed` (随机种子), `experiment.device` (计算设备 'cuda' 或 'cpu'), `hydra.run.dir` (输出路径)。
    *   **数据流**: Hydra 将所有配置整合成一个 `DictConfig` 对象 (`cfg`) 传递给 `main` 函数。

2.  **数据加载 (`EnhancedDataLoader`)**
    *   **目的**: 从指定路径加载原始或处理过的数据文件。
    *   **操作**: 根据配置决定是加载完整数据集还是迷你数据集 (`use_mini_dataset`)，并从对应路径 (`paths.raw_data_dir` 或 `paths.mini_data_dir`) 读取目标场 (`target_field`)、协变量 (`covariate_fields`) 和掩码 (`mask_path`) 文件。支持多种文件格式（.npy, .nc, .mat 等）和缓存 (`use_cache`)。
    *   **关键参数**: `data.dataset.*`, `paths.*`, `data.preprocessing.apply_mask`, `data.sequence.include_covariates`。
    *   **数据流**: 文件系统中的数据文件 -> NumPy 数组字典 (`data = {'salinity': ..., 'wind_u': ..., 'mask': ...}`)

3.  **数据预处理 (`DataProcessor.process`)**
    *   **目的**: 对加载的数据进行标准化/归一化、掩码应用和数据集划分。
    *   **操作**: 
        *   根据配置 (`standardization` 或 `normalize`) 对目标场和协变量进行缩放（使用训练集数据拟合缩放器以防数据泄露）。
        *   如果 `apply_mask` 为 `true`，则将掩码应用于目标场数据。
        *   根据 `split_ratio` 将数据按时间顺序划分为训练集、验证集和测试集。
    *   **关键参数**: `data.preprocessing.*`, `data.dataset.split_ratio`。
    *   **数据流**: 原始 NumPy 数组字典 -> 经过缩放、掩码处理并划分好的 NumPy 数组字典 (`processed_data = {'train_target': ..., 'val_target': ..., 'train_covariates': ...}`)

4.  **创建自编码器数据集 (`DataProcessor.create_ae_datasets`)**
    *   **目的**: 将预处理后的数据转换为适用于自编码器训练的 PyTorch 数据集。
    *   **操作**: 将训练、验证、测试集的目标场数据 (`*_target`) 转换为 PyTorch `TensorDataset`。
    *   **数据流**: NumPy 数组 -> PyTorch `TensorDataset` (每个样本是单个时间步的盐度场)。

5.  **自编码器 (AE) 训练或加载 (`train_autoencoder`, `AutoencoderModel`)**
    *   **目的**: 训练或加载一个预训练的自编码器模型用于降维。
    *   **操作**: 
        *   初始化 `AutoencoderModel` (结构由 `model.autoencoder.*` 定义)。
        *   如果 `model.ae_lstm.use_pretrained_ae` 为 `true` 且预训练模型存在于 `paths.ae_model_dir`，则加载模型权重。
        *   否则，调用 `train_autoencoder` 函数，使用 AE 数据集和 `train.autoencoder.*` 配置进行训练。
    *   **关键参数**: `model.autoencoder.*`, `train.autoencoder.*`, `model.ae_lstm.use_pretrained_ae`, `paths.ae_model_dir`。
    *   **数据流**: AE 数据集 -> 训练过程 -> 训练好的 `AutoencoderModel`。

6.  **生成潜在表示 (`DataProcessor.generate_latent_representations`)**
    *   **目的**: 使用训练好的 AE 将高维的盐度场数据编码为低维潜在向量。
    *   **操作**: 将训练、验证、测试集的 AE 数据集输入到 AE 的编码器部分，得到对应的潜在空间表示。
    *   **数据流**: AE 数据集 + 训练好的 AE -> 低维 NumPy 数组 (`train_latent`, `val_latent`, `test_latent`)。

7.  **创建序列数据集 (`DataProcessor.create_sequence_datasets`)**
    *   **目的**: 将潜在表示和协变量（如果使用）构造成适用于 LSTM 训练的时间序列样本。
    *   **操作**: 
        *   使用滑动窗口方法，根据 `sequence_length` 和 `prediction_horizon` 从潜在表示和协变量数据中提取输入序列 (X) 和目标序列 (Y)。
        *   将序列数据转换为 PyTorch `TensorDataset`。
    *   **关键参数**: `data.sequence.*`。
    *   **数据流**: 低维 NumPy 数组 (latent codes, covariates) -> PyTorch `TensorDataset` (每个样本是 `(序列输入, 序列目标)`)。

8.  **LSTM 训练或加载 (`train_lstm`, `LSTMModel`)**
    *   **目的**: 训练或加载一个预训练的 LSTM 模型用于学习潜在空间中的时间动态。
    *   **操作**: 
        *   初始化 `LSTMModel` (结构由 `model.lstm.*` 定义，`input_size` 需匹配 AE 的 `latent_dim`)。
        *   如果 `model.ae_lstm.use_pretrained_lstm` 为 `true` 且预训练模型存在于 `paths.lstm_model_dir`，则加载模型权重。
        *   否则，调用 `train_lstm` 函数，使用序列数据集和 `train.lstm.*` 配置进行训练。
    *   **关键参数**: `model.lstm.*`, `train.lstm.*`, `model.ae_lstm.use_pretrained_lstm`, `paths.lstm_model_dir`。
    *   **数据流**: 序列数据集 -> 训练过程 -> 训练好的 `LSTMModel`。

9.  **组合 AE-LSTM 模型与可选微调 (`AELSTMModel`, `train_ae_lstm`)**
    *   **目的**: 将训练好的 AE 和 LSTM 组合起来，并可选择进行端到端微调。
    *   **操作**: 
        *   初始化 `AELSTMModel`，它内部包含 AE 和 LSTM。
        *   如果 `model.ae_lstm.train_ae_end_to_end` 为 `true`，则调用 `train_ae_lstm` 函数，使用 **AE 数据集** (注意：输入是原始高维数据) 和 `train.ae_lstm.*` 配置进行端到端训练（同时调整 AE 和 LSTM 的权重）。
    *   **关键参数**: `model.ae_lstm.train_ae_end_to_end`, `train.ae_lstm.*`。
    *   **数据流**: 训练好的 AE + 训练好的 LSTM -> `AELSTMModel`。如果微调，则 AE 数据集 -> 微调过程 -> 更新后的 `AELSTMModel`。

10. **模型评估 (`evaluate_model`)**
    *   **目的**: 在测试集上评估训练好的模型的性能。
    *   **操作**: 
        *   分别调用 `evaluate_model` 评估 AE 的重建性能（使用 `test_dataset_ae`）和组合模型 (`AELSTMModel`) 的预测性能（使用 `test_dataset_lstm`）。
        *   计算 MAE, RMSE, R² 等指标。
    *   **数据流**: 测试数据集 + 训练好的模型 -> 包含指标和预测结果的字典 (`ae_metrics`, `lstm_metrics`)。
    *   **返回值**: 对于 Optuna 优化，`main` 函数会返回 `lstm_metrics['metrics']['rmse']` 作为优化的目标值。

11. **结果可视化 (`plot_*` 函数)**
    *   **目的**: 生成可视化图表以分析模型性能。
    *   **操作**: 调用 `src/visualization/visualize.py` 中的函数，如 `plot_reconstruction_samples`, `plot_prediction_samples`, `plot_metrics`，将评估结果和配置 (`visualization.*`) 作为输入，生成并保存图像到 `paths.visualization_dir`。
    *   **关键参数**: `visualization.*`, `paths.visualization_dir`。
    *   **数据流**: 评估结果字典 + 测试数据 -> 图像文件。

12. **日志记录与 TensorBoard (`logging`, `SummaryWriter`)**
    *   **贯穿流程**: 在整个流程中，使用 Python `logging` 模块记录关键信息和进度。
    *   **TensorBoard**: 如果 `logging.use_tensorboard` 为 `true`，则初始化 `SummaryWriter` 并将训练/验证损失、指标等写入 TensorBoard 日志文件 (位于 `paths.tensorboard_dir`)。

## 使用方法

### 基本用法

```bash
python main_pipeline.py
```

### 覆盖配置

```bash
# 使用不同的模型参数
python main_pipeline.py model=custom_model

# 更改训练参数
python main_pipeline.py train.epochs=100 train.batch_size=32

# 更改数据源
python main_pipeline.py data.dataset.use_mini_dataset=True
```

### 超参数优化

```bash
# 使用 Optuna 运行超参数搜索
python main_pipeline.py --multirun train.optimizer.lr=tag(log,interval(1e-4,1e-2)) model.ae.latent_dim=choice(16,32,64)
```

## 依赖项

完整的依赖列表请参见 `requirements.txt` 文件。

## 开发

要扩展此项目：

1. 在 `src/model/` 中添加新的模型架构
2. 在 `conf/` 中添加相应的配置
3. 在 `src/data/` 中添加新的数据集类型
4. 在 `src/visualization/` 中添加新的可视化功能
