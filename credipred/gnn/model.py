from typing import Any, Type, Union

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
    GraphTransformerModule,
    LabelPredictor,
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
        'GT': GraphTransformerModule,
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
        binary: bool,
        **kwargs: Any,
    ):
        super().__init__()
        self.model_name = model_name
        self.is_gps = model_name == 'GPS'
        self.is_gt = model_name == 'GT'
        self.binary = binary
        normalization_cls = self.normalization_map[normalization]
        self.input_linear = nn.Linear(
            in_features=in_channels, out_features=hidden_channels
        )
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

        self.re_modules = nn.ModuleList()

        for _ in range(num_layers):
            if self.is_gps:
                self.re_modules.append(
                    GraphGPSResidualWrapper(
                        normalization=normalization_cls,
                        dim=hidden_channels,
                        dropout=dropout,
                        heads=kwargs['gps_head'],
                        attn_type=kwargs['gps_attn_type'],
                        local_mpnn_type=kwargs['gps_local_mpnn'],
                    )
                )
            elif self.is_gt:
                self.re_modules.append(
                    GraphTransformerModule(
                        dim=hidden_channels,
                        dropout=dropout,
                    )
                )
            else:
                self.re_modules.append(
                    ResidualModuleWrapper(
                        module=self.modules[model_name],
                        normalization=normalization_cls,
                        dim=hidden_channels,
                        dropout=dropout,
                    )
                )

        self.output_normalization = normalization_cls(hidden_channels)
        self.output_linear = nn.Linear(
            in_features=hidden_channels, out_features=out_channels
        )
        self.node_predictor = NodePredictor(in_dim=out_channels, out_dim=1)
        self.label_predictor = LabelPredictor(in_dim=out_channels, out_dim=2)

    def forward(
        self, x: Tensor, edge_index: Tensor | None = None, batch: Tensor | None = None
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
        x = self.output_linear(x)
        if not self.binary:
            x = self.node_predictor(x)
        else:
            x = self.label_predictor(x)
        return x

    def get_embeddings(
        self, x: Tensor, edge_index: Tensor | None = None, batch: Tensor | None = None
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
        x = self.output_linear(x)
        return x
