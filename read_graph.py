import os
import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB, Actor

root = os.path.split(__file__)[0]

def load_dataset(name: str, device=None):
    if device is None:
        device = torch.device('cpu')
    name = name.lower()
    if name in ["cora", "pubmed", "citeseer"]:
        dataset = Planetoid(root=root + "/dataset/Planetoid", name=name)
    elif name in ["chameleon", "squirrel"]:
        dataset = WikipediaNetwork(root=root + "/dataset/WikipediaNetwork", name=name)
    elif name in ["cornell", "texas", "wisconsin"]:
        dataset = WebKB(root=root + "/dataset/WebKB", name=name)
    elif name in ["actor"]:
        dataset = Actor(root=root + "/dataset/Actor")
    else:
        raise "Please implement support for this dataset in function load_dataset()."
    data = dataset[0].to(device)
    x, y = data.x, data.y
    n = len(x)
    edge_index = data.edge_index
    nclass = len(torch.unique(y))
    # return eidx_to_sp(n, edge_index), x, y, nclass, split_mask(data.train_mask), split_mask(data.val_mask), split_mask(data.test_mask)
    return eidx_to_sp(n, edge_index), x, y, nclass

def eidx_to_sp(n: int, edge_index: torch.Tensor, device=None) -> torch.sparse.Tensor:
    indices = edge_index
    values = torch.FloatTensor([1.0] * len(edge_index[0])).to(edge_index.device)
    coo = sp.coo_matrix((values, (indices[0, :], indices[1, :])), shape=[n ,n])
    # coo = torch.sparse_coo_tensor(indices=indices, values=values, size=[n, n])

    # mean_nodes = average_node_degree(coo)
    # print(mean_nodes)

    # build symmetric adjacency matrix, but not self edge
    coo = coo + coo.T.multiply(coo.T > coo) - coo.multiply(coo.T > coo)
    if device is None:
        device = edge_index.device
    return coo.tocoo()

def split_mask(data_mask: torch.Tensor):
    if data_mask.ndim > 1:
        return [data_mask[:, i] for i in range(data_mask.shape[1])]
    return [data_mask]

def average_node_degree(adjacency_matrix):
    num_nodes = adjacency_matrix.shape[0]
    num_edges = adjacency_matrix.nnz  # 统计非零元素数量，即边数
    average_degree = num_edges / num_nodes
    return average_degree
