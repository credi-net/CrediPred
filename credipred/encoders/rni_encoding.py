import torch
from torch import Tensor

from tgrag.encoders.encoder import Encoder


class RNIEncoder(Encoder):
    def __init__(self, dimension: int):
        self.dimension = dimension

    def __call__(self, length: int) -> Tensor:
        return torch.rand(length, self.dimension)
