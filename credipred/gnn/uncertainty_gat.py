"""Uncertainty-weighted GAT: modifies attention weights by source node confidence.

attention_ij = softmax(e_ij) * confidence_j  (then re-normalize)

This down-weights messages from uncertain (wide-interval) neighbors,
addressing the heterophily-induced coverage failure in Q1 nodes.
"""

from typing import Type, Union

import torch
from torch import Tensor, nn
from torch_geometric.nn import GATConv
from torch_geometric.utils import softmax as pyg_softmax


NormalizationType = Union[Type[nn.Identity], Type[nn.LayerNorm], Type[nn.BatchNorm1d]]


class UncertaintyGATConv(nn.Module):
    """GAT convolution that re-weights attention by source-node confidence."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.conv = GATConv(dim, dim, add_self_loops=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(dim // 2, dim),
            nn.Dropout(p=dropout),
        )

    def forward(
        self, x: Tensor, edge_index: Tensor, confidence: Tensor | None = None,
    ) -> Tensor:
        if confidence is None:
            # Fall back to standard GAT
            x = self.conv(x, edge_index)
            return x + self.feed_forward(x)

        # Get attention weights from GATConv
        out, (ei, alpha) = self.conv(
            x, edge_index, return_attention_weights=True,
        )
        # alpha: [num_edges, 1], ei: [2, num_edges] (includes self-loops)
        src_nodes = ei[0]
        dst_nodes = ei[1]

        # Multiply attention by source confidence and re-normalize
        src_conf = confidence[src_nodes].unsqueeze(-1)  # [E, 1]
        weighted_alpha = alpha * src_conf
        weighted_alpha = pyg_softmax(weighted_alpha, dst_nodes)

        # Manual aggregation: x_src * weighted_alpha, scatter to dst
        lin = self.conv.lin_src if self.conv.lin_src is not None else self.conv.lin
        x_transformed = lin(x)
        msg = x_transformed[src_nodes] * weighted_alpha
        out = torch.zeros_like(x)
        out.scatter_add_(0, dst_nodes.unsqueeze(-1).expand_as(msg), msg)

        return out + self.feed_forward(out)


class UncertaintyResidualWrapper(nn.Module):
    """Residual wrapper that passes confidence through."""

    def __init__(
        self,
        module: nn.Module,
        normalization: NormalizationType,
        dim: int,
    ):
        super().__init__()
        self.normalization = normalization(dim)
        self.module = module

    def forward(
        self, x: Tensor, edge_index: Tensor, confidence: Tensor | None = None,
    ) -> Tensor:
        x_res = self.normalization(x)
        x_res = self.module(x_res, edge_index, confidence=confidence)
        return x + x_res


class UncertaintyGATModel(nn.Module):
    """Full model with uncertainty-weighted GAT layers."""

    normalization_map: dict[str, NormalizationType] = {
        'none': nn.Identity,
        'LayerNorm': nn.LayerNorm,
        'BatchNorm': nn.BatchNorm1d,
    }

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        prediction_dim: int = 3,
        normalization: str = 'BatchNorm',
    ):
        super().__init__()
        normalization_cls = self.normalization_map[normalization]

        self.input_linear = nn.Linear(in_channels, hidden_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = UncertaintyGATConv(dim=hidden_channels, dropout=dropout)
            wrapper = UncertaintyResidualWrapper(
                module=conv,
                normalization=normalization_cls,
                dim=hidden_channels,
            )
            self.layers.append(wrapper)

        self.output_normalization = normalization_cls(hidden_channels)
        self.output_linear = nn.Linear(hidden_channels, out_channels)

        # Predictor head: outputs prediction_dim values with sigmoid
        hidden_pred = max(out_channels // 2, 1)
        self.predictor = nn.Sequential(
            nn.Linear(out_channels, hidden_pred),
            nn.ReLU(),
            nn.Linear(hidden_pred, prediction_dim),
            nn.Sigmoid(),
        )

    def forward(
        self, x: Tensor, edge_index: Tensor, confidence: Tensor | None = None,
    ) -> Tensor:
        x = self.input_linear(x)
        x = self.dropout(x)
        x = self.act(x)

        for layer in self.layers:
            x = layer(x, edge_index, confidence=confidence)

        x = self.output_normalization(x)
        x = self.output_linear(x)
        x = self.predictor(x)
        return x
