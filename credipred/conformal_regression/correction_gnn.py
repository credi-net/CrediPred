"""Post-hoc topology correction GNN for quantile regression.

Given frozen base-model predictions [mid, lower, upper] per node,
a small GNN propagates these along the graph and predicts an additive
residual delta.  The corrected prediction is:

    corrected = base + delta

Trained with pinball loss so the correction can tighten/shift intervals
by exploiting neighbourhood structure.
"""

from typing import Type, Union

import torch
from torch import Tensor, nn
from torch_geometric.nn import GATConv


NormalizationType = Union[Type[nn.Identity], Type[nn.LayerNorm], Type[nn.BatchNorm1d]]


class CorrectionGATConv(nn.Module):
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
        return x + self.feed_forward(x)  # residual connection within the block, before outer normalization


class CorrectionResidualWrapper(nn.Module):
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


class CorrectionGNN(nn.Module):
    """Topology correction model for quantile regression.

    Input:  base predictions [N, 3] (mid, lower, upper)
    Output: delta [N, 3] added to base predictions

    The model maps 3-dim input into a hidden space, applies several
    GAT layers with residual connections, then projects back to 3-dim.
    Output is *unbounded* (no sigmoid) so it can shift predictions
    in either direction.

    Args:
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
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        normalization: str = 'BatchNorm',
    ):
        super().__init__()
        normalization_cls = self.normalization_map[normalization]
        pred_dim = 3  # mid, lower, upper

        self.input_linear = nn.Linear(pred_dim, hidden_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = CorrectionGATConv(dim=hidden_channels, dropout=dropout)
            wrapper = CorrectionResidualWrapper(
                module=conv,
                normalization=normalization_cls,
                dim=hidden_channels,
            )
            self.layers.append(wrapper)

        self.output_normalization = normalization_cls(hidden_channels)
        self.output_linear = nn.Linear(hidden_channels, pred_dim)

        # Initialize output near zero so initial correction is small
        nn.init.zeros_(self.output_linear.weight)
        nn.init.zeros_(self.output_linear.bias)

    def forward(self, base_preds: Tensor, edge_index: Tensor) -> Tensor:
        """Predict additive delta given base predictions and graph structure.

        Args:
            base_preds: [N, 3] frozen base model predictions.
            edge_index: [2, E] graph edges.

        Returns:
            delta: [N, 3] correction to add to base_preds.
        """
        x = self.input_linear(base_preds)
        x = self.dropout(x)
        x = self.act(x)

        for layer in self.layers:
            x = layer(x, edge_index)

        x = self.output_normalization(x)
        delta = self.output_linear(x)
        return delta
