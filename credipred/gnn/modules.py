from typing import Type, Union

from torch import Tensor, nn
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, GINConv, SAGEConv

NormalizationType = Union[Type[nn.Identity], Type[nn.LayerNorm], Type[nn.BatchNorm1d]]


class NodePredictor(nn.Module):
    """Node prediction head with configurable activation.

    Args:
        in_dim: Input dimension
        hidden_dim_multiplier: Multiplier for hidden dimension
        out_dim: Output dimension
        predictor_type: Type of predictor head
            - 'relu_sigmoid': Original (ReLU + Sigmoid) - has issues with extreme values
            - 'leaky_sigmoid': LeakyReLU + Sigmoid - preserves negative information
            - 'pure_sigmoid': GELU + Sigmoid - smooth activation, strict [0,1] output
            - 'no_sigmoid': GELU only - no output constraint, best gradient flow
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim_multiplier: float = 0.5,
        out_dim: int = 1,
        predictor_type: str = 'relu_sigmoid',
    ):
        super().__init__()
        self.predictor_type = predictor_type
        hidden_dim = int(hidden_dim_multiplier * in_dim)
        self.lin_node = nn.Linear(in_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin_node(x)

        if self.predictor_type == 'relu_sigmoid':
            # Original: ReLU + Sigmoid (has issues with extreme values)
            x = x.relu()
            x = self.out(x)
            x = x.sigmoid()
        elif self.predictor_type == 'leaky_sigmoid':
            # LeakyReLU preserves negative information + Sigmoid
            x = nn.functional.leaky_relu(x, negative_slope=0.1)
            x = self.out(x)
            x = x.sigmoid()
        elif self.predictor_type == 'pure_sigmoid':
            # GELU (smooth, preserves negatives) + Sigmoid
            x = nn.functional.gelu(x)
            x = self.out(x)
            x = x.sigmoid()
        elif self.predictor_type == 'no_sigmoid':
            # GELU only, no sigmoid - best gradient flow
            x = nn.functional.gelu(x)
            x = self.out(x)
            # No sigmoid, output is unconstrained
        else:
            raise ValueError(f"Unknown predictor_type: {self.predictor_type}")

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
