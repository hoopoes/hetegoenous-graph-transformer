import os
import gc

import torch

from torch_geometric.data import HeteroData
from torch_geometric.datasets import DBLP
from torch_geometric.loader import NeighborLoader, HGTLoader, NeighborSampler

from utils import *


# Pilot Test file
path = os.path.join(os.getcwd(), 'data/DBLP')
dataset = DBLP(path)
data = dataset[0]
# Let's change the data for simplicity
# leave only author and paper
hetero_data = HeteroData()

# create two node types "author" and "paper" holding a feature matrix
hetero_data['author'].x = data['author'].x
hetero_data['author'].train_mask = torch.full((data['author'].x.shape[0], ), True)

# add train_mask (necessary for train/test split)
hetero_data['paper'].x = data['paper'].x
hetero_data['paper'].train_mask = torch.full((data['paper'].x.shape[0], ), True)
# Create an edge type and build the graph connectivity
# shape: (2, num_edges)
# Let's say our edge type is ('paper', 'written_by', 'author')
# then edge_index should look like
# [[paper, paper, paper, ...],
#  [author, author, author, ...]]
# so the 1st row of edge_index is source node
# and 2nd row of edge_index is target node (flow: source_to_target)
hetero_data['paper', 'written_by', 'author'].edge_index = data['paper', 'to', 'author']['edge_index']

del data
_ = gc.collect()
print(hetero_data)


# Neighbor Sampler
from typing import Union, List, Dict, Callable, Optional, Any
from torch_geometric.typing import EdgeType, InputNodes

from collections.abc import Sequence

import torch

from torch_geometric.data import HeteroData
from torch_geometric.loader.base import BaseDataLoader
from torch_geometric.loader.utils import edge_type_to_str
from torch_geometric.loader.utils import to_hetero_csc
from torch_geometric.loader.utils import filter_hetero_data

NumNeighbors = Union[List[int], Dict[EdgeType, List[int]]]

out = to_hetero_csc(hetero_data, device='cpu')
colptr_dict, row_dict, perm_dict = out

node_types, edge_types = hetero_data.metadata()
num_neighbors = {key: [11] for key in edge_types}

num_neighbors = {
    edge_type_to_str(key): value
    for key, value in num_neighbors.items()
}
# {'paper__written_by__author': [10]}

num_hops = max([len(v) for v in num_neighbors.values()])    # 1
input_node_type = 'author'

indices = [0, 0, 10, 10, 20, 20]
index = torch.tensor(indices)

sample_fn = torch.ops.torch_sparse.hetero_neighbor_sample
node_dict, row_dict, col_dict, edge_dict = sample_fn(
    node_types,
    edge_types,
    colptr_dict,
    row_dict,
    {input_node_type: index},
    num_neighbors,
    num_hops,
    False,
    True)

# ns = NeighborSampler(hetero_data, [10], False, False, 'author')
new_data = filter_hetero_data(hetero_data, node_dict, row_dict, col_dict, edge_dict, perm_dict)

edge_index = hetero_data[('paper', 'written_by', 'author')]['edge_index']

new_data['paper', 'written_by', 'author']['edge_index']




