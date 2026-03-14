from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn


class DenseGATLayer(nn.Module):
    """Dense multi-head GAT layer for inputs shaped [B, N, F]."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int, dropout: float, residual: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual and in_dim == out_dim

        self.q_proj = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.k_proj = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.v_proj = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.out_proj = nn.Linear(out_dim * num_heads, out_dim)

        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F], attn_mask: [B, N, N] (True for valid attention pairs), node_mask: [B, N]
        bsz, n_nodes, _ = x.shape
        q = self.q_proj(x).view(bsz, n_nodes, self.num_heads, self.out_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, n_nodes, self.num_heads, self.out_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, n_nodes, self.num_heads, self.out_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.out_dim)  # [B, H, N, N]
        pair_mask = attn_mask.unsqueeze(1) & node_mask.unsqueeze(1).unsqueeze(-1)
        scores = scores.masked_fill(~pair_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = self.drop(attn)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, n_nodes, self.num_heads * self.out_dim)
        out = self.out_proj(out)
        out = self.drop(out)
        if self.residual:
            out = out + x
        out = self.norm(out)
        out = self.act(out)
        out = out * node_mask.unsqueeze(-1)
        return out


class GATQM9Regressor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        residual: bool,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            DenseGATLayer(hidden_dim, hidden_dim, num_heads=num_heads, dropout=dropout, residual=residual)
            for _ in range(num_layers)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, node_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        h = h * node_mask.unsqueeze(-1)
        for layer in self.layers:
            h = layer(h, attn_mask, node_mask)

        # masked mean pooling: [B, N, H] -> [B, H]
        denom = node_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        graph_emb = (h * node_mask.unsqueeze(-1)).sum(dim=1) / denom
        pred = self.head(graph_emb)
        return pred, graph_emb
