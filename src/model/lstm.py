#!/usr/bin/env python
# src/model/lstm.py - LSTM model for time series prediction

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Union, Any
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    """
    LSTM model for time series prediction.
    """
    def __init__(self, cfg: DictConfig):
        super(LSTMModel, self).__init__()
        self.cfg = cfg
        
        # LSTM parameters
        self.input_size = cfg.model.lstm.input_size # 这应该是潜在变量的维度
        self.hidden_size = cfg.model.lstm.hidden_size
        self.num_layers = cfg.model.lstm.num_layers
        self.output_size = cfg.model.lstm.output_size # 这通常与 input_size 相同，如果LSTM预测的是下一个潜在状态
        self.dropout = cfg.model.lstm.dropout
        
        # Layer Normalization for the input
        # LayerNorm 会对最后一个维度（input_size/特征维度）进行归一化
        self.input_layernorm = nn.LayerNorm(self.input_size) 
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True, # batch_first=True 表示输入和输出张量的形状为 (batch, seq, feature)
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        # Optional: Layer Normalization for the LSTM output's hidden states
        self.output_layernorm = nn.LayerNorm(self.hidden_size)

        # Output layer
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]: # 注意: 原始返回类型包含 hidden，如果只返回 out，可以简化
        """
        Forward pass through LSTM.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
            hidden: Optional hidden state
            
        Returns:
            Output tensor and optional hidden state
        """
        x_normalized = self.input_layernorm(x)
        
        lstm_out, hidden_state_out = self.lstm(x_normalized, hidden) 
        
        out = self.fc(lstm_out[:, -1, :])

        return out 
    
    def predict_sequence(
        self, 
        initial_sequence: torch.Tensor, 
        steps: int
    ) -> torch.Tensor:
        """
        Generate multi-step predictions.
        
        Args:
            initial_sequence: Initial sequence to start prediction [batch_size, seq_len, input_size]
            steps: Number of steps to predict
            
        Returns:
            Predicted sequence [batch_size, steps, output_size]
        """
        self.eval()
        device = next(self.parameters()).device
        batch_size = initial_sequence.size(0)
        
        # Clone the input sequence to avoid modifying it
        current_sequence = initial_sequence.clone()
        predictions = torch.zeros(batch_size, steps, self.output_size).to(device)
        
        # Hidden state
        h = None
        
        with torch.no_grad():
            # Make predictions one step at a time
            for i in range(steps):
                # Generate prediction from current sequence
                output, h = self.lstm(current_sequence, h)
                output = self.fc(output[:, -1, :])
                
                # Store prediction
                predictions[:, i, :] = output
                
                # Update sequence for next prediction
                # Remove first entry and add prediction as last entry
                current_sequence = torch.cat([
                    current_sequence[:, 1:, :],
                    output.unsqueeze(1)
                ], dim=1)
        
        return predictions


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM model for time series prediction.
    """
    def __init__(self, cfg: DictConfig):
        super(BidirectionalLSTM, self).__init__()
        self.cfg = cfg
        
        # Model parameters
        self.input_size = cfg.model.lstm.input_size
        self.hidden_size = cfg.model.lstm.hidden_size
        self.num_layers = cfg.model.lstm.num_layers
        self.output_size = cfg.model.lstm.output_size
        self.dropout = cfg.model.lstm.dropout
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        # Output layer (x2 for bidirectional)
        self.fc = nn.Linear(self.hidden_size * 2, self.output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through bidirectional LSTM.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
            
        Returns:
            Output tensor of shape [batch_size, output_size]
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Get prediction from last time step
        out = self.fc(lstm_out[:, -1, :])
        
        return out


class AttentionLSTM(nn.Module):
    """
    LSTM model with attention mechanism.
    """
    def __init__(self, cfg: DictConfig):
        super(AttentionLSTM, self).__init__()
        self.cfg = cfg
        
        # Model parameters
        self.input_size = cfg.model.lstm.input_size
        self.hidden_size = cfg.model.lstm.hidden_size
        self.num_layers = cfg.model.lstm.num_layers
        self.output_size = cfg.model.lstm.output_size
        self.dropout = cfg.model.lstm.dropout
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1)
        )
        
        # Output layer
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attention LSTM.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
            
        Returns:
            Output tensor of shape [batch_size, output_size]
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size]
        
        # Calculate attention weights
        attention_weights = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention
        context = torch.sum(attention_weights * lstm_out, dim=1)  # [batch_size, hidden_size]
        
        # Output layer
        out = self.fc(context)
        
        return out


def get_lstm_model(cfg: DictConfig) -> nn.Module:
    """
    Factory function to create appropriate LSTM model.
    
    Args:
        cfg: Configuration object
    
    Returns:
        LSTM model instance
    """
    lstm_type = cfg.model.lstm.get("type", "standard").lower()
    
    if lstm_type == "standard":
        return LSTMModel(cfg)
    elif lstm_type == "bidirectional":
        return BidirectionalLSTM(cfg)
    elif lstm_type == "attention":
        return AttentionLSTM(cfg)
    else:
        log.warning(f"Unknown LSTM type: {lstm_type}, using standard")
        return LSTMModel(cfg)
