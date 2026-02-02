from typing import Any, Dict, Optional, Type, Union

import torch
from torch import Tensor, nn

from credipred.gnn.modules import (
    FFModule,
    GATModule,
    GATv2Module,
    GCNModule,
    GINModule,
    GraphGPSModule,
    GraphGPSResidualWrapper,
    NodePredictor,
    ResidualModuleWrapper,
    SAGEModule,
)

NormalizationType = Union[Type[nn.Identity], Type[nn.LayerNorm], Type[nn.BatchNorm1d]]


class Model(torch.nn.Module):
    modules: dict[str, torch.nn.Module] = {
        'GCN': GCNModule,
        'SAGE': SAGEModule,
        'GAT': GATModule,
        'GATv2': GATv2Module,
        'GIN': GINModule,
        'FF': FFModule,
        'GPS': GraphGPSModule,
    }
    normalization_map: dict[str, NormalizationType] = {
        'none': torch.nn.Identity,
        'LayerNorm': torch.nn.LayerNorm,
        'BatchNorm': torch.nn.BatchNorm1d,
    }

    def __init__(
        self,
        model_name: str,
        normalization: str,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        # GraphGPS specific parameters
        gps_heads: int = 4,
        gps_attn_type: str = 'multihead',
        gps_attn_kwargs: Optional[Dict[str, Any]] = None,
        gps_local_mpnn: str = 'gin',
    ):
        super().__init__()
        self.model_name = model_name
        self.is_gps = model_name == 'GPS'
        normalization_cls = self.normalization_map[normalization]
        self.input_linear = nn.Linear(
            in_features=in_channels, out_features=hidden_channels
        )
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

        self.re_modules = nn.ModuleList()

        for _ in range(num_layers):
            if self.is_gps:
                # Use GraphGPS-specific residual wrapper
                residual_module = GraphGPSResidualWrapper(
                    normalization=normalization_cls,
                    dim=hidden_channels,
                    dropout=dropout,
                    heads=gps_heads,
                    attn_type=gps_attn_type,
                    attn_kwargs=gps_attn_kwargs,
                    local_mpnn_type=gps_local_mpnn,
                )
            else:
                residual_module = ResidualModuleWrapper(
                    module=self.modules[model_name],
                    normalization=normalization_cls,
                    dim=hidden_channels,
                    dropout=dropout,
                )
            self.re_modules.append(residual_module)

        self.output_normalization = normalization_cls(hidden_channels)
        self.output_linear = nn.Linear(
            in_features=hidden_channels, out_features=out_channels
        )
        self.node_predictor = NodePredictor(in_dim=out_channels, out_dim=1)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor | None = None,
        batch: Tensor | None = None,
    ) -> Tensor:
        x = self.input_linear(x)
        x = self.dropout(x)
        x = self.act(x)

        for re_module in self.re_modules:
            if self.is_gps:
                # GraphGPS needs batch tensor for global attention
                x = re_module(x, edge_index, batch)
            elif edge_index is not None:
                x = re_module(x, edge_index)
            else:
                x = re_module(x)

        x = self.output_normalization(x)
        x = self.output_linear(x)
        x = self.node_predictor(x)
        return x

    def get_embeddings(
        self,
        x: Tensor,
        edge_index: Tensor | None = None,
        batch: Tensor | None = None,
    ) -> Tensor:
        x = self.input_linear(x)
        x = self.dropout(x)
        x = self.act(x)

        for re_module in self.re_modules:
            if self.is_gps:
                x = re_module(x, edge_index, batch)
            elif edge_index is not None:
                x = re_module(x, edge_index)
            else:
                x = re_module(x)

        x = self.output_normalization(x)
        return x
