import copy
import torch
import numpy as np
import scipy.sparse as sp
from math import sin, cos
import networkx as nx
import random
from read_graph import load_dataset
from torch_cluster import random_walk
from torch_sparse import SparseTensor
from collections import defaultdict
from sklearn.metrics import f1_score
import requests
import json

def generate_random_walk(adj, nodes: torch.Tensor, p, q) -> torch.Tensor:
    long_walks_per_node = 6
    long_walk_len = 8
    batch = nodes
    batch = batch.repeat(long_walks_per_node, 1).t()
    batch = batch.reshape(-1)
    rowptr = torch.LongTensor(adj.row)
    col = torch.LongTensor(adj.col)
    rw = random_walk(rowptr, col, batch, long_walk_len, p, q)
    rw = rw.reshape(nodes.shape[0], -1)
    return rw

def ope(adj):
    if sp.isspmatrix_csr(adj):
        hop_1 = adj.toarray()
    else:
        hop_1 = adj
    I = np.eye(hop_1.shape[0])
    hop_1[np.where(hop_1 > 0)] = 1
    hop_1 = hop_1 - I
    hop_1[np.where(hop_1 < 0)] = 0
    return hop_1

def multi_hop_adj(adj, hop):
    adj_list = [adj.tocsr()]
    uni_hop_list = [adj.toarray()]

    for i in range(1, hop):
        adj_list.append(adj_list[i-1] @ adj)
        uni_hop_list.append(ope(ope(adj_list[i]) - ope(adj_list[i-1])))

    return uni_hop_list

# 计算节点之间的最短距离
def cal_hop_adj(adj):
    csr_adj = sp.csr_matrix(adj)
    hop_adj = sp.csgraph.floyd_warshall(csgraph=csr_adj, directed=False, return_predecessors=False)
    hop_adj = torch.from_numpy(hop_adj)
    hop_adj[torch.isinf(hop_adj)] = 999999
    return torch.tensor(hop_adj, dtype=torch.int64)


def laplacian_positional_encoding(adj, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    L = sp.csgraph.laplacian(adj)

    # Eigenvectors with scipy
    #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    lap_pos_enc = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float()
    re_lap_pos_enc = re_features(lap_pos_enc)

    return re_lap_pos_enc

# 计算相对位置编码，t表示相对位置
def caL_relative_pe(t: list, xdim):
    pe_dim = list(range(xdim//2 + 1))
    w_list = list(map(lambda x: (10000**(2*x/xdim)), pe_dim))
    pe_list = []
    for ti in t:
        relative_pe = [torch.tensor([sin(wt*ti), cos(wt*ti)]).t() for wt in w_list]
        relative_pe = torch.cat(relative_pe, dim=0)
        pe_list.append(relative_pe.unsqueeze(0))
    pe = torch.cat(pe_list, dim=0)
    pe_split = torch.split(pe, xdim, dim=1)
    return pe_split[0]

def cal_accuracy(output, labels):
    preds = torch.argmax(output, dim=1)
    correct = preds.eq(labels)
    correct = correct.sum().item()
    return correct

def generate_relative_pe(subgra: torch.Tensor, hop_adj: torch.Tensor, relative_pe_fea: torch.Tensor):
    relative_hop = [hop_adj[x[0], x].unsqueeze(0) for x in subgra.tolist()]
    relative_hop = torch.cat(relative_hop, dim=0)
    relative_pe = relative_pe_fea[relative_hop]
    return relative_pe


def hop_nodes_dict(adj):
    a = np.where(adj == 1)
    hop_1_dict = defaultdict(list)
    for k, v in zip(a[0], a[1]):
        hop_1_dict[k].append(v)

    return hop_1_dict

def hierarchy_sampling(b_data, hop_dict_list, sample_n):
    random.seed(123)
    sampling_nods = []
    for index in range(b_data.shape[0]):
        nodes = [int(b_data[index])]
        for hop in hop_dict_list:
            if sample_n > len(hop[nodes[0]]):
                nodes += hop[nodes[0]] + [-1]*(sample_n - len(hop[nodes[0]]))
            else:
                nodes += random.sample(hop[nodes[0]], sample_n)
        sampling_nods.append(nodes)

    sampling_nods = torch.tensor(sampling_nods)

    return sampling_nods

def re_features(features):
    padding = torch.zeros((1, features.shape[1]))
    re_features = torch.cat((features, padding), dim=0)

    return re_features

def cal_average_time(time_l):
    time_l.remove(max(time_l))
    time_l.remove(min(time_l))
    return np.mean(time_l)

def load_sp(data_name):
    file_name = './dataset_split/' + data_name + '.pt'
    data_ten = torch.load(file_name)
    return data_ten.numpy().tolist()
