import torch
from tgrag.encoders.rni_encoding import RNIEncoder


def test_rni_call():
    rni = RNIEncoder(4)
    x = rni(5)
    assert isinstance(x, torch.Tensor)
    print(x)
    print(x.shape)
    assert x.shape == torch.Size([5, 4])
