# -*- coding: utf-8 -*-
"""Common neural layers used by PILOT and optional internal references."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) // 2
        x = x.permute(0, 2, 1)
        x = torch.cat([x[:, :, :1].repeat(1, 1, pad), x, x[:, :, -1:].repeat(1, 1, pad)], dim=-1)
        return self.avg(x).permute(0, 2, 1)


class NLinear(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x[:, :, :self.n_channels]
        last = v[:, -1:, :]
        z = (v - last).permute(0, 2, 1)
        out = self.linear(z).permute(0, 2, 1)
        return out + last


class DLinear(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, n_channels: int, kernel_size: int = 25):
        super().__init__()
        self.n_channels = n_channels
        self.ma = MovingAverage(kernel_size)
        self.ls = nn.Linear(seq_len, pred_len)
        self.lt = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x[:, :, :self.n_channels]
        trend = self.ma(v)
        seasonal = v - trend
        out = self.ls(seasonal.permute(0, 2, 1)) + self.lt(trend.permute(0, 2, 1))
        return out.permute(0, 2, 1)


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, groups: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=self.pad, dilation=dilation, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.pad > 0:
            y = y[:, :, :-self.pad]
        return y


class ModernTCNBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.dw = CausalConv1d(dim, dim, kernel_size=kernel_size, dilation=dilation, groups=dim)
        self.pw1 = nn.Conv1d(dim, dim * 2, 1)
        self.pw2 = nn.Conv1d(dim * 2, dim, 1)
        self.norm = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        y = self.dw(x)
        y = self.norm(y)
        y = F.gelu(self.pw1(y))
        y = self.drop(self.pw2(y))
        return res + y


class ModernTCN(nn.Module):
    def __init__(self, input_dim: int, pred_len: int, n_channels: int, hidden: int = 128, layers: int = 5, dropout: float = 0.1):
        super().__init__()
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.inp = nn.Conv1d(input_dim, hidden, 1)
        blocks = [ModernTCNBlock(hidden, kernel_size=5, dilation=2 ** min(i, 4), dropout=dropout) for i in range(layers)]
        self.net = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, pred_len * n_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        h = self.net(self.inp(z))[:, :, -1]
        return self.head(h).view(x.size(0), self.pred_len, self.n_channels)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerForecast(nn.Module):
    def __init__(self, input_dim: int, pred_len: int, n_channels: int, d_model: int = 128, heads: int = 4, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=256, dropout=dropout, batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, pred_len * n_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(self.pos(self.proj(x)))[:, -1]
        return self.head(h).view(x.size(0), self.pred_len, self.n_channels)
