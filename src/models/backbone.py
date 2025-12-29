"""Transformer骨架網絡模塊

提供基於Transformer的特征提取器，用於處理序列數據（如Mel頻譜圖）
"""
import math

import torch
from torch import nn


def build_sinusoidal_positions(length, dim, device):
    """構建正弦位置編碼

    Args:
        length: 序列長度
        dim: 嵌入維度
        device: 計算設備

    Returns:
        正弦位置編碼矩陣，形狀為 (length, dim)
    """
    position = torch.arange(length, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TransformerBackbone(nn.Module):
    """Transformer骨架網絡

    用於提取序列數據的高級特征表示，基於PyTorch的TransformerEncoder

    Args:
        input_dim: 輸入特征維度
        d_model: 模型嵌入維度
        n_heads: 注意力頭數量
        n_layers: Transformer層數
        ff_mult: 前饋網絡隱藏層擴展倍數
        dropout: Dropout概率
    """
    def __init__(
        self,
        input_dim,
        d_model=256,
        n_heads=4,
        n_layers=4,
        ff_mult=4,
        dropout=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        # 輸入投影層，將輸入特征映射到模型嵌入維度
        self.input_proj = nn.Linear(input_dim, d_model)
        # 構建Transformer編碼器層
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
        )
        # 構建多層Transformer編碼器
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        """前向傳播

        Args:
            x: 輸入特征，形狀為 (batch_size, seq_length, input_dim)
            padding_mask: 填充掩碼，形狀為 (batch_size, seq_length)，值為True的位置表示填充

        Returns:
            提取的特征，形狀為 (batch_size, seq_length, d_model)
        """
        # 輸入投影
        x = self.input_proj(x)
        # 添加位置編碼
        pos = build_sinusoidal_positions(x.size(1), self.d_model, x.device)
        x = x + pos.unsqueeze(0)
        # 應用Dropout
        x = self.dropout(x)
        # 通過Transformer編碼器
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return x
