import torch
from torch_geometric.utils import to_undirected


def test_undirected_equal():
    directed_edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

    directed_edge_index = directed_edge_index.contiguous()

    undirected_edge_index = to_undirected(directed_edge_index)

    print("Directed")
    print(directed_edge_index)
    print("Undirected")
    print(undirected_edge_index)

def test_undirected_small():
    directed_edge_index = torch.tensor([[1, 1], [0, 2]], dtype=torch.long)

    directed_edge_index = directed_edge_index.contiguous()

    undirected_edge_index = to_undirected(directed_edge_index)

    print("Directed")
    print(directed_edge_index)
    print("Undirected")
    print(undirected_edge_index)
