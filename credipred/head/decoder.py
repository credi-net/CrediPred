import torch
from torch import Tensor
from torch.nn import LayerNorm, Linear


class NodePredictor(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.lin_node = Linear(in_dim, hidden_dim)
        self.out = Linear(hidden_dim, out_dim)

    def reset_parameters(self) -> None:
        self.lin_node.reset_parameters()
        self.out.reset_parameters()

    def forward(self, node_embedding: Tensor) -> Tensor:
        h = self.lin_node(node_embedding)
        h = h.relu()
        h = self.out(h)
        h = h.sigmoid()
        return h


class LabelPredictor(torch.nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim_multiplier: float = 0.5, out_dim: int = 2
    ):
        super().__init__()
        self.input_norm = LayerNorm(in_dim)
        hidden_dim = int(hidden_dim_multiplier * in_dim)  # hidden_dim=64
        self.lin_node = Linear(in_dim, hidden_dim)
        self.out = Linear(hidden_dim, out_dim)
        self.activation = torch.nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_norm(x)
        x = self.lin_node(x)
        x = self.activation(x)
        x = self.out(x)
        return torch.log_softmax(x, dim=-1)

    def predict(self, x: Tensor) -> Tensor:
        return self.forward(x.float()).argmax(dim=-1)
