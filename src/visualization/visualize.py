#!/usr/bin/env python
# src/visualization/visualize.py - Visualization utilities for model outputs

import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from omegaconf import DictConfig
import io
from PIL import Image

log = logging.getLogger(__name__)

def fig_to_rgb_array(fig: plt.Figure) -> np.ndarray:
    """将 Matplotlib Figure 对象转换为 RGB NumPy 数组 (H, W, C)。"""
    with io.BytesIO() as buf:
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=fig.get_dpi())
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        arr = np.array(img)
    return arr # 返回 (H, W, C) 格式的数组

def load_triangulation_data(cfg: DictConfig) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Loads vertices and triangles for tripcolor plotting."""
    # 构建 vertices 和 triangles 文件的完整路径
    # 默认情况下，这些文件应该位于 cfg.paths.data_dir 中
    # 文件名本身可以从 cfg.data.vertices_file 和 cfg.data.triangles_file 获取
    
    base_data_dir = cfg.paths.get('data_dir', '.') # 如果未设置，默认为当前目录
    
    vertices_filename = cfg.data.get('vertices_file', 'vertices.npy')
    triangles_filename = cfg.data.get('triangles_file', 'triangles.npy')

    # 确保我们处理的是绝对路径或相对于 data_dir 的路径
    if os.path.isabs(vertices_filename):
        vertices_path = vertices_filename
    else:
        vertices_path = os.path.join(base_data_dir, vertices_filename)

    if os.path.isabs(triangles_filename):
        triangles_path = triangles_filename
    else:
        triangles_path = os.path.join(base_data_dir, triangles_filename)

    log.info(f"Attempting to load vertices from: {vertices_path}")
    log.info(f"Attempting to load triangles from: {triangles_path}")

    vertices, triangles = None, None
    try:
        if os.path.exists(vertices_path):
            vertices = np.load(vertices_path, allow_pickle=True) # 允许pickle以防旧格式
            log.info(f"Loaded vertices from {vertices_path}, shape: {vertices.shape}")
        else:
            log.error(f"Vertices file not found at {vertices_path}")

        if os.path.exists(triangles_path):
            triangles = np.load(triangles_path, allow_pickle=True) # 允许pickle以防旧格式
            log.info(f"Loaded triangles from {triangles_path}, shape: {triangles.shape}")
        else:
            log.error(f"Triangles file not found at {triangles_path}")
            
    except Exception as e:
        log.error(f"Error loading triangulation data: {e}")
        return None, None # 确保在出错时返回 None
    
    if vertices is None or triangles is None:
        log.warning("Failed to load vertices or triangles. Plotting will not be possible.")
        return None, None

    # 处理顶点数据 (确保是 (N, 2) 形状)
    if vertices.ndim == 2 and vertices.shape[0] == 2 and vertices.shape[1] > 0: # 检查是否是 (2, N)
        log.info(f"Transposing vertices from {vertices.shape} to ({vertices.shape[1]}, {vertices.shape[0]}) for plotting.")
        vertices = vertices.T
    elif vertices.ndim == 2 and vertices.shape[1] == 2: # 已经是 (N, 2)
        log.info(f"Vertices shape is already suitable: {vertices.shape}")
    else:
        log.error(f"Vertices are not in the expected (N, 2) or (2, N) shape. Got shape: {vertices.shape}.")
        return None, triangles # 返回 None 表示顶点数据有问题

    # 处理三角面片数据 (确保是 (M, 3) 形状并且是基于0的索引)
    if triangles.ndim == 2 and triangles.shape[1] == 3:
        if np.issubdtype(triangles.dtype, np.number):
            # 检查是否需要减1 (通常网格数据可能是基于1的索引)
            # 这里假设如果最小值大于0，则需要减1
            if triangles.min() > 0:
                 log.info("Triangles min value > 0, assuming 1-based indexing and subtracting 1.")
                 triangles = (triangles - 1).astype(np.int_)
            else:
                 triangles = triangles.astype(np.int_) # 确保是整数类型
            log.info(f"Triangles after processing: shape {triangles.shape}, dtype {triangles.dtype}")
        else:
            log.error(f"Triangles data type is not numeric ({triangles.dtype}).")
            return vertices, None
    else:
        log.error(f"Triangles are not in the expected (M, 3) shape. Got shape: {triangles.shape}.")
        return vertices, None
        
    return vertices, triangles

def plot_tripcolor_field(
    cfg: DictConfig,
    field_values: np.ndarray,
    title: str,
    colorbar_label: str,
    ax: Optional[plt.Axes] = None,
    cmap: str = 'viridis',
    shading: str = 'gouraud',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    save_path: Optional[str] = None
) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
    """
    基础绘图函数，用于在三角网格上绘制指定场的值。

    Args:
        cfg: Hydra 配置对象。
        field_values: 要绘制的场值数组，形状应为 (num_vertices,)。
        title: 图像标题。
        colorbar_label: 颜色条的标签。
        ax: 可选的 Matplotlib Axes 对象，在其上绘图。如果为 None，则创建新的 Figure 和 Axes。
        cmap: Matplotlib colormap 名称。
        shading: 'flat' 或 'gouraud'。
        vmin: 颜色映射的最小值。
        vmax: 颜色映射的最大值。
        save_path: 可选，保存图像的路径。如果提供，则保存图像并关闭。

    Returns:
        Tuple[Optional[plt.Figure], Optional[plt.Axes]]
        如果提供了 ax，则返回 (None, ax)。
        如果未提供 ax，则返回 (fig, ax)。
        如果绘图失败，则返回 (None, None)。
    """
    vertices, triangles = load_triangulation_data(cfg)

    if vertices is None or triangles is None:
        log.error("Cannot plot field: Missing or invalid vertices or triangles data.")
        return None, None

    if field_values.shape[0] != vertices.shape[0]:
        log.error(
            f"Field values array shape ({field_values.shape}) "
            f"does not match number of vertices ({vertices.shape[0]}). Cannot plot."
        )
        return None, None

    fig_created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=cfg.visualization.get('figsize', [10, 8]))
        fig_created = True
    else:
        fig = ax.get_figure()

    # 自动确定 vmin 和 vmax (如果未提供)
    current_vmin = vmin if vmin is not None else field_values.min()
    current_vmax = vmax if vmax is not None else field_values.max()

    tpc = ax.tripcolor(
        vertices[:, 0], vertices[:, 1], triangles, field_values,
        cmap=cmap, shading=shading, vmin=current_vmin, vmax=current_vmax
    )

    ax.set_title(title, fontsize=cfg.visualization.plot_settings.get('title_fontsize', 14))
    ax.set_xlabel("Longitude", fontsize=cfg.visualization.plot_settings.get('axis_label_fontsize', 12))
    ax.set_ylabel("Latitude", fontsize=cfg.visualization.plot_settings.get('axis_label_fontsize', 12))
    ax.tick_params(axis='both', which='major', labelsize=cfg.visualization.plot_settings.get('tick_label_fontsize', 10))
    ax.set_aspect('equal') # 保持地理数据的纵横比

    if cfg.visualization.plot_settings.get('grid', True):
        ax.grid(True, linestyle='--', alpha=0.7)

    # 添加颜色条
    if cfg.visualization.plot_settings.get('colorbar', True):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(tpc, cax=cax)
        cbar.set_label(colorbar_label, fontsize=cfg.visualization.plot_settings.get('axis_label_fontsize', 12))
        cbar.ax.tick_params(labelsize=cfg.visualization.plot_settings.get('tick_label_fontsize', 10))

    if fig_created: # 只有当这个函数创建了 figure 时才调整布局和保存
        plt.tight_layout()
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, dpi=cfg.visualization.get('dpi', 300))
                log.info(f"Field plot saved to {save_path}")
                plt.close(fig) # 保存后关闭图像
                return None, None # 表示已保存并关闭
            except Exception as e:
                log.error(f"Error saving plot to {save_path}: {e}")
        return fig, ax
    else: # 如果 ax 是传入的，则不关闭，由调用者处理
        return None, ax


def plot_salinity_field_psu(
    cfg: DictConfig,
    salinity_values: np.ndarray,
    title_prefix: str = "Salinity Field",
    ax: Optional[plt.Axes] = None,
    vmin: Optional[float] = None, # 通常盐度有其典型范围，例如 0-35 PSU
    vmax: Optional[float] = None,
    save_path: Optional[str] = None
) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
    """绘制盐度场 (单位: PSU)。"""
    title = f"{title_prefix}"
    colorbar_label = "Salinity (PSU)"
    # 您可能想为盐度图指定一个特定的颜色映射，例如 'ocean_r' 或 'viridis'
    # vmin, vmax 可以根据数据的典型范围预设或从数据中动态计算
    return plot_tripcolor_field(cfg, salinity_values, title, colorbar_label, ax=ax,
                                cmap=cfg.visualization.get('cmap_salinity', 'viridis'),
                                vmin=vmin, vmax=vmax, save_path=save_path)


def plot_correlation_field(
    cfg: DictConfig,
    correlation_values: np.ndarray, # 假设是每个顶点与某个参考点的时间序列相关性系数
    title_prefix: str = "Correlation Field",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None
) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
    """绘制单点相关性系数场。相关系数通常在 -1 到 1 之间。"""
    title = f"{title_prefix}"
    colorbar_label = "Correlation Coefficient"
    # 对于相关性，发散的颜色映射（如 'coolwarm', 'RdBu_r'）通常更好
    return plot_tripcolor_field(cfg, correlation_values, title, colorbar_label, ax=ax,
                                cmap=cfg.visualization.get('cmap_correlation', 'coolwarm'),
                                vmin=-1.0, vmax=1.0, save_path=save_path)


def plot_spatial_rmse_field_psu(
    cfg: DictConfig,
    rmse_values: np.ndarray, # 每个空间点的 RMSE 值
    title_prefix: str = "Spatial RMSE Distribution",
    ax: Optional[plt.Axes] = None,
    vmin: Optional[float] = 0.0, # RMSE >= 0
    vmax: Optional[float] = None,
    save_path: Optional[str] = None
) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
    """绘制空间 RMSE 分布 (单位: PSU)。"""
    title = f"{title_prefix}"
    colorbar_label = "RMSE (PSU)"
    # RMSE 通常使用顺序颜色映射，例如 'magma' 或 'plasma'
    return plot_tripcolor_field(cfg, rmse_values, title, colorbar_label, ax=ax,
                                cmap=cfg.visualization.get('cmap_rmse', 'magma'),
                                vmin=vmin, vmax=vmax, save_path=save_path)


def plot_error_field_psu(
    cfg: DictConfig,
    error_values: np.ndarray, # 例如 真实值 - 重建值
    title_prefix: str = "Error Field (True - Reconstructed)",
    ax: Optional[plt.Axes] = None,
    vmin: Optional[float] = None, # 误差可正可负
    vmax: Optional[float] = None,
    save_path: Optional[str] = None
) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
    """绘制误差场 (真实 - 重建) (单位: PSU)。"""
    title = f"{title_prefix}"
    colorbar_label = "Error (PSU)"
    # 误差场通常使用发散的颜色映射，中心为0，例如 'RdBu_r'
    # 动态计算 vmin 和 vmax 使0居中，或根据误差范围设定
    abs_max_error = np.abs(error_values).max()
    dynamic_vmin = -abs_max_error if vmin is None else vmin
    dynamic_vmax = abs_max_error if vmax is None else vmax

    return plot_tripcolor_field(cfg, error_values, title, colorbar_label, ax=ax,
                                cmap=cfg.visualization.get('cmap_error', 'RdBu_r'),
                                vmin=dynamic_vmin, vmax=dynamic_vmax, save_path=save_path)


# --- 原有的其他绘图函数可以保留或根据新的基础函数进行调整 ---

def plot_reconstruction_samples(
    model: torch.nn.Module,
    test_loader, # 假设 test_loader 提供的是原始高维数据
    cfg: DictConfig,
    device: torch.device,
    n_samples: int = 5,
    save_dir: Optional[str] = None # 修改为 save_dir
):
    """
    Plots original vs. reconstructed samples using tripcolor for triangular meshes.
    """
    model.eval()
    # vertices, triangles = load_triangulation_data(cfg) # 现在由 plot_tripcolor_field 内部加载

    # if vertices is None or triangles is None:
    #     log.error("Cannot plot reconstruction samples: Missing vertices or triangles data.")
    #     return

    if save_dir is None:
        save_dir = os.path.join(cfg.paths.visualization_dir, "reconstructions")
    os.makedirs(save_dir, exist_ok=True)

    samples_collected = 0
    
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            if samples_collected >= n_samples:
                break
            
            # 假设 test_loader 的每个 batch_data 是 [inputs_tensor]
            # 如果 test_loader 返回的是 (data, target) 元组，则取 batch_data[0]
            if isinstance(batch_data, list) or isinstance(batch_data, tuple):
                inputs = batch_data[0].to(device)
            else: # 假设直接是输入张量
                inputs = batch_data.to(device)

            if inputs.ndim == 1: # 如果是单个样本，增加 batch 维度
                inputs = inputs.unsqueeze(0)
            
            reconstructions = model(inputs) # model 是自编码器

            for j in range(inputs.size(0)): # 遍历 batch 中的每个样本
                if samples_collected >= n_samples:
                    break

                original_values = inputs[j].cpu().numpy()
                reconstructed_values = reconstructions[j].cpu().numpy()

                # 确保值是一维数组
                if original_values.ndim > 1: original_values = original_values.squeeze()
                if reconstructed_values.ndim > 1: reconstructed_values = reconstructed_values.squeeze()
                
                # 确定颜色范围
                vmin_val = min(original_values.min(), reconstructed_values.min())
                vmax_val = max(original_values.max(), reconstructed_values.max())

                # 绘制原始样本
                plot_salinity_field_psu(
                    cfg,
                    original_values,
                    title_prefix=f"Original Sample {samples_collected + 1}",
                    vmin=vmin_val,
                    vmax=vmax_val,
                    save_path=os.path.join(save_dir, f"sample_{samples_collected+1}_original.png")
                )

                # 绘制重建样本
                plot_salinity_field_psu(
                    cfg,
                    reconstructed_values,
                    title_prefix=f"Reconstructed Sample {samples_collected + 1}",
                    vmin=vmin_val,
                    vmax=vmax_val,
                    save_path=os.path.join(save_dir, f"sample_{samples_collected+1}_reconstructed.png")
                )
                
                # 绘制误差图
                error_values = original_values - reconstructed_values
                plot_error_field_psu(
                    cfg,
                    error_values,
                    title_prefix=f"Error (Orig - Recon) Sample {samples_collected + 1}",
                    # vmin, vmax 会被 plot_error_field_psu 内部动态调整以使0居中
                    save_path=os.path.join(save_dir, f"sample_{samples_collected+1}_error.png")
                )
                
                samples_collected += 1
    
    if samples_collected > 0:
        log.info(f"{samples_collected} reconstruction (tripcolor) sample plots saved to {save_dir}")
    else:
        log.warning("No samples were collected/plotted for reconstruction.")


def plot_prediction_samples(
    model: torch.nn.Module, # 这个 model 应该是 AE-LSTM 或 LSTM+AE解码器
    test_loader: DataLoader, 
    cfg: DictConfig,
    device: torch.device,
    autoencoder_model: Optional[torch.nn.Module] = None, # 如果 model 是纯 LSTM，则需要 AE 解码
    n_samples: int = 5,
    save_dir: Optional[str] = None
):
    """
    Plots true future values vs. predicted future values using tripcolor.
    Assumes model output and targets are feature vectors corresponding to vertices.
    """
    model.eval()
    if autoencoder_model:
        autoencoder_model.eval()

    if save_dir is None:
        save_dir = os.path.join(cfg.paths.visualization_dir, "predictions")
    os.makedirs(save_dir, exist_ok=True)

    samples_collected = 0
    
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader): # test_loader 提供 (input_sequence, target_values)
            if samples_collected >= n_samples:
                break
            
            inputs_seq, true_values_at_vertices = batch_data[0].to(device), batch_data[1].to(device)
            
            # 获取预测值
            if autoencoder_model: # 如果是纯 LSTM + AE 模式
                latent_predictions = model(inputs_seq) # LSTM 输出潜在表示
                predicted_values_at_vertices = autoencoder_model.decode(latent_predictions)
            else: # 如果是端到端的 AE-LSTM 模型
                predicted_values_at_vertices = model(inputs_seq)

            for j in range(true_values_at_vertices.size(0)): # Iterate through batch
                if samples_collected >= n_samples:
                    break

                true_sample_flat = true_values_at_vertices[j].cpu().numpy()
                predicted_sample_flat = predicted_values_at_vertices[j].cpu().numpy()

                if true_sample_flat.ndim > 1: true_sample_flat = true_sample_flat.squeeze()
                if predicted_sample_flat.ndim > 1: predicted_sample_flat = predicted_sample_flat.squeeze()
                
                vmin_val = min(true_sample_flat.min(), predicted_sample_flat.min())
                vmax_val = max(true_sample_flat.max(), predicted_sample_flat.max())

                # 绘制真实值
                plot_salinity_field_psu(
                    cfg,
                    true_sample_flat,
                    title_prefix=f"True Future Sample {samples_collected + 1}",
                    vmin=vmin_val,
                    vmax=vmax_val,
                    save_path=os.path.join(save_dir, f"sample_{samples_collected+1}_true_future.png")
                )
                
                # 绘制预测值
                plot_salinity_field_psu(
                    cfg,
                    predicted_sample_flat,
                    title_prefix=f"Predicted Future Sample {samples_collected + 1}",
                    vmin=vmin_val,
                    vmax=vmax_val,
                    save_path=os.path.join(save_dir, f"sample_{samples_collected+1}_predicted_future.png")
                )
                
                # 绘制预测误差
                error_values = true_sample_flat - predicted_sample_flat
                plot_error_field_psu(
                    cfg,
                    error_values,
                    title_prefix=f"Prediction Error (True - Pred) Sample {samples_collected + 1}",
                    save_path=os.path.join(save_dir, f"sample_{samples_collected+1}_prediction_error.png")
                )
                
                samples_collected += 1

    if samples_collected > 0:
        log.info(f"{samples_collected} prediction (tripcolor) sample plots saved to {save_dir}")
    else:
        log.warning("No samples were collected/plotted for prediction.")


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    cfg: DictConfig,
    model_type: str = "model" # 改为通用 model_type
) -> None:
    """
    Plot training and validation loss curves.
    """
    output_dir = os.path.join(cfg.paths.visualization_dir, "loss_curves")
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=cfg.visualization.get('figsize', [10, 6]))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Training Loss', marker='o', linestyle='-')
    ax.plot(epochs, val_losses, label='Validation Loss', marker='x', linestyle='--')
    
    ax.set_xlabel('Epochs', fontsize=cfg.visualization.plot_settings.get('axis_label_fontsize', 12))
    ax.set_ylabel('Loss', fontsize=cfg.visualization.plot_settings.get('axis_label_fontsize', 12))
    ax.set_title(f'{model_type.upper()} Training & Validation Loss', fontsize=cfg.visualization.plot_settings.get('title_fontsize', 14))
    ax.legend(fontsize=cfg.visualization.plot_settings.get('legend_fontsize', 10))
    ax.grid(True, alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=cfg.visualization.plot_settings.get('tick_label_fontsize', 10))
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{cfg.experiment.name}_{model_type}_loss_curves.png")
    fig.savefig(save_path, dpi=cfg.visualization.get('dpi', 300))
    plt.close(fig)
    
    log.info(f"Saved {model_type} loss curves to {save_path}")

# metrics_plots 函数可以保持不变，或者根据需要调整
def plot_metrics(
    metrics_dict: Dict[str, Dict[str, Any]], # 例如: {'Autoencoder': ae_metrics, 'AE-LSTM': lstm_metrics}
    cfg: DictConfig,
    title_suffix: str = "Performance Metrics"
) -> None:
    """
    Plot evaluation metrics for different models or stages.
    
    Args:
        metrics_dict: Dictionary where keys are model names/stages and values are metric dicts.
                      Example: {'Autoencoder': {'rmse': 0.1, 'mae': 0.05}, 
                                'AE-LSTM': {'rmse': 0.5, 'mae': 0.3}}
        cfg: Configuration object.
        title_suffix: Suffix for plot titles.
    """
    output_dir = os.path.join(cfg.paths.visualization_dir, "metrics_comparison")
    os.makedirs(output_dir, exist_ok=True)

    model_names = list(metrics_dict.keys())
    if not model_names:
        log.warning("No metrics provided to plot_metrics.")
        return

    # 收集所有出现过的指标名称
    all_metric_keys = set()
    for model_name in model_names:
        if metrics_dict[model_name]: # 确保指标字典不为空
            all_metric_keys.update(metrics_dict[model_name].keys())
    
    if not all_metric_keys:
        log.warning("No metric keys found in the provided metrics_dict.")
        return

    num_metrics = len(all_metric_keys)
    num_models = len(model_names)

    # 为每个指标创建一个条形图
    for metric_key in sorted(list(all_metric_keys)):
        values = []
        valid_model_names_for_metric = []
        for model_name in model_names:
            if metrics_dict[model_name] and metric_key in metrics_dict[model_name]:
                values.append(metrics_dict[model_name][metric_key])
                valid_model_names_for_metric.append(model_name)
        
        if not values: # 如果这个指标没有任何模型有值，则跳过
            continue

        fig, ax = plt.subplots(figsize=cfg.visualization.get('figsize_bar', [8, 6]))
        bars = ax.bar(valid_model_names_for_metric, values, color=plt.cm.get_cmap('viridis', len(values))(np.linspace(0, 1, len(values))))
        
        ax.set_ylabel(metric_key.upper(), fontsize=cfg.visualization.plot_settings.get('axis_label_fontsize', 12))
        ax.set_title(f'{metric_key.upper()} - {title_suffix}', fontsize=cfg.visualization.plot_settings.get('title_fontsize', 14))
        ax.set_xticklabels(valid_model_names_for_metric, rotation=45, ha="right", fontsize=cfg.visualization.plot_settings.get('tick_label_fontsize', 10))
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        # 在条形图上显示数值
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center', fontsize=cfg.visualization.plot_settings.get('tick_label_fontsize', 9))
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{cfg.experiment.name}_{metric_key}_comparison.png")
        fig.savefig(save_path, dpi=cfg.visualization.get('dpi', 300))
        plt.close(fig)
        log.info(f"Saved {metric_key} comparison plot to {save_path}")

    log.info(f"All metrics comparison plots saved to {output_dir}")