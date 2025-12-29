import math

import torch
from torch import nn


def build_sinusoidal_positions(length, dim, device):
    position = torch.arange(length, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TransformerBackbone(nn.Module):
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
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        x = self.input_proj(x)
        pos = build_sinusoidal_positions(x.size(1), self.d_model, x.device)
        x = x + pos.unsqueeze(0)
        x = self.dropout(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return x
