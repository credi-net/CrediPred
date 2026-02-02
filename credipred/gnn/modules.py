from typing import Any, Dict, Optional, Type, Union

from torch import Tensor, nn
from torch_geometric.nn import (
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    GPSConv,
    ResGatedGraphConv,
    SAGEConv,
)

NormalizationType = Union[Type[nn.Identity], Type[nn.LayerNorm], Type[nn.BatchNorm1d]]


class NodePredictor(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim_multiplier: float = 0.5, out_dim: int = 1
    ):
        super().__init__()
        hidden_dim = int(hidden_dim_multiplier * in_dim)
        self.lin_node = nn.Linear(in_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin_node(x)
        x = x.relu()
        x = self.out(x)
        x = x.sigmoid()
        return x


class ResidualModuleWrapper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        normalization: NormalizationType,
        dim: int,
        dropout: float,
    ):
        super().__init__()
        self.normalization = normalization(dim)
        self.module = module(
            dim=dim,
            dropout=dropout,
        )

    def forward(self, x: Tensor, edge_index: Tensor | None = None) -> Tensor:
        x_res = self.normalization(x)
        if edge_index is not None:
            x_res = self.module(x_res, edge_index)
        else:
            x_res = self.module(x_res)
        x = x + x_res
        return x


class FeedForwardModule(nn.Module):
    def __init__(
        self,
        dim: int,
        dropout: float,
        hidden_channel_multipler: float = 0.5,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(
            in_features=dim,
            out_features=int(hidden_channel_multipler * dim),
        )
        self.dropout_1 = nn.Dropout(p=dropout)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(
            in_features=int(hidden_channel_multipler * dim),
            out_features=dim,
        )
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.linear_1(x)
        x = self.dropout_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)

        return x


class GCNModule(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.conv = GCNConv(dim, dim)
        self.feed_forward_module = FeedForwardModule(
            dim=dim,
            dropout=dropout,
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv(x, edge_index)
        x = self.feed_forward_module(x, edge_index)
        return x


class SAGEModule(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.conv = SAGEConv(dim, dim)
        self.feed_forward_module = FeedForwardModule(
            dim=dim,
            dropout=dropout,
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv(x, edge_index)
        x = self.feed_forward_module(x, edge_index)
        return x


class GATModule(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.conv = GATConv(dim, dim)
        self.feed_forward_module = FeedForwardModule(
            dim=dim,
            dropout=dropout,
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv(x, edge_index)
        x = self.feed_forward_module(x, edge_index)
        return x


class GATv2Module(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.conv = GATv2Conv(dim, dim)
        self.feed_forward_module = FeedForwardModule(
            dim=dim,
            dropout=dropout,
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv(x, edge_index)
        x = self.feed_forward_module(x, edge_index)
        return x


class GINModule(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.conv = GINConv(mlp)
        self.feed_forward_module = FeedForwardModule(
            dim=dim,
            dropout=dropout,
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv(x, edge_index)
        x = self.feed_forward_module(x, edge_index)
        return x


class FFModule(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.feed_forward_module = FeedForwardModule(
            dim=dim,
            dropout=dropout,
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.feed_forward_module(x)
        return x


class GraphGPSModule(nn.Module):
    """GraphGPS layer combining local message passing with global attention.

    This module wraps PyTorch Geometric's GPSConv which combines:
    - Local MPNN (message passing neural network): GIN, GatedGCN, or GAT
    - Global attention mechanism (multihead or performer)

    Reference: https://arxiv.org/abs/2205.12454
    """

    def __init__(
        self,
        dim: int,
        dropout: float,
        heads: int = 4,
        attn_type: str = 'multihead',
        attn_kwargs: Optional[Dict[str, Any]] = None,
        local_mpnn_type: str = 'gin',
    ):
        super().__init__()
        if attn_kwargs is None:
            attn_kwargs = {'dropout': dropout}

        # Select local MPNN based on type
        if local_mpnn_type == 'gin':
            # GIN with MLP (original GPS default)
            local_nn = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
            )
            local_mpnn = GINConv(local_nn)
        elif local_mpnn_type == 'gatedgcn':
            # GatedGCN - recommended by GraphGPS paper
            local_mpnn = ResGatedGraphConv(dim, dim)
        elif local_mpnn_type == 'gat':
            # GAT - local attention mechanism
            local_mpnn = GATConv(dim, dim, heads=1, concat=False)
        else:
            raise ValueError(
                f'Unknown local_mpnn_type: {local_mpnn_type}. '
                "Choices: 'gin', 'gatedgcn', 'gat'."
            )

        # GPSConv combines local MPNN + global attention
        self.conv = GPSConv(
            channels=dim,
            conv=local_mpnn,
            heads=heads,
            attn_type=attn_type,
            attn_kwargs=attn_kwargs,
            dropout=dropout,
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with optional batch tensor for global attention.

        Args:
            x: Node features [num_nodes, dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment for each node [num_nodes] (optional)

        Returns:
            Updated node features [num_nodes, dim]
        """
        x = self.conv(x, edge_index, batch=batch)
        return x


class GraphGPSResidualWrapper(nn.Module):
    """Residual wrapper specifically for GraphGPS that handles batch parameter."""

    def __init__(
        self,
        normalization: NormalizationType,
        dim: int,
        dropout: float,
        heads: int = 4,
        attn_type: str = 'multihead',
        attn_kwargs: Optional[Dict[str, Any]] = None,
        local_mpnn_type: str = 'gin',
    ):
        super().__init__()
        self.normalization = normalization(dim)
        self.module = GraphGPSModule(
            dim=dim,
            dropout=dropout,
            heads=heads,
            attn_type=attn_type,
            attn_kwargs=attn_kwargs,
            local_mpnn_type=local_mpnn_type,
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        x_res = self.normalization(x)
        x_res = self.module(x_res, edge_index, batch)
        x = x + x_res
        return x
