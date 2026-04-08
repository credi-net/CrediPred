"""Simple ConfGNN correction model for classification.

Following the original CF-GNN (snap-stanford/conformalized-gnn):
- Input: frozen base softmax probabilities [N, num_classes]
- A small GNN propagates these along the graph
- Output: corrected logits [N, num_classes]
- corrected = base_softmax + delta

No structure learning — just topology-aware correction.
"""

from typing import Type, Union

import torch
from torch import Tensor, nn
from torch_geometric.nn import GATConv


NormalizationType = Union[Type[nn.Identity], Type[nn.LayerNorm], Type[nn.BatchNorm1d]]


class ConfGATConv(nn.Module):
    """Single GAT conv block with feed-forward for the correction network."""

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

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv(x, edge_index)
        return x + self.feed_forward(x)


class ConfResidualWrapper(nn.Module):
    """Pre-norm residual wrapper."""

    def __init__(
        self,
        module: nn.Module,
        normalization: NormalizationType,
        dim: int,
    ):
        super().__init__()
        self.normalization = normalization(dim)
        self.module = module

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x_res = self.normalization(x)
        x_res = self.module(x_res, edge_index)
        return x + x_res


class ConfGNN(nn.Module):
    """Topology correction model for classification conformal prediction.

    Input:  base softmax probabilities [N, num_classes]
    Output: delta [N, num_classes] added to base probabilities

    Args:
        num_classes: number of output classes (2 for binary).
        hidden_channels: hidden dimension for GAT layers.
        num_layers: number of GAT conv layers.
        dropout: dropout rate.
        normalization: one of 'none', 'LayerNorm', 'BatchNorm'.
    """

    normalization_map: dict[str, NormalizationType] = {
        'none': nn.Identity,
        'LayerNorm': nn.LayerNorm,
        'BatchNorm': nn.BatchNorm1d,
    }

    def __init__(
        self,
        num_classes: int = 2,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        normalization: str = 'BatchNorm',
    ):
        super().__init__()
        normalization_cls = self.normalization_map[normalization]

        self.input_linear = nn.Linear(num_classes, hidden_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = ConfGATConv(dim=hidden_channels, dropout=dropout)
            wrapper = ConfResidualWrapper(
                module=conv,
                normalization=normalization_cls,
                dim=hidden_channels,
            )
            self.layers.append(wrapper)

        self.output_normalization = normalization_cls(hidden_channels)
        self.output_linear = nn.Linear(hidden_channels, num_classes)

        # Initialize output near zero so initial correction is small
        nn.init.zeros_(self.output_linear.weight)
        nn.init.zeros_(self.output_linear.bias)

    def forward(self, base_softmax: Tensor, edge_index: Tensor) -> Tensor:
        """Predict additive delta given base softmax and graph structure.

        Args:
            base_softmax: [N, num_classes] frozen base model softmax.
            edge_index: [2, E] graph edges.

        Returns:
            delta: [N, num_classes] correction to add to base_softmax.
        """
        x = self.input_linear(base_softmax)
        x = self.dropout(x)
        x = self.act(x)

        for layer in self.layers:
            x = layer(x, edge_index)

        x = self.output_normalization(x)
        delta = self.output_linear(x)
        return delta
