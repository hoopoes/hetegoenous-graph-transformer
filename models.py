import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.inits import glorot, ones, reset
from torch_geometric.typing import EdgeType, Metadata, NodeType
from torch_geometric.utils import softmax


def group(xs: List[Tensor], aggr: Optional[str]) -> Optional[Tensor]:
    if len(xs) == 0:
        return None
    elif aggr is None:
        return torch.stack(xs, dim=1)
    elif len(xs) == 1:
        return xs[0]
    else:
        out = torch.stack(xs, dim=0)
        out = getattr(torch, aggr)(out, dim=0)
        out = out[0] if isinstance(out, tuple) else out
        return out


class HGTConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        group: str = "sum",
        use_edge_type_params: bool = False,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.group = group
        self.use_edge_type_params = use_edge_type_params

        self.k_lin = torch.nn.ModuleDict()
        self.q_lin = torch.nn.ModuleDict()
        self.v_lin = torch.nn.ModuleDict()
        self.a_lin = torch.nn.ModuleDict()
        self.skip = torch.nn.ParameterDict()

        for node_type, in_channels in self.in_channels.items():
            self.k_lin[node_type] = Linear(in_channels, out_channels)
            self.q_lin[node_type] = Linear(in_channels, out_channels)
            self.v_lin[node_type] = Linear(in_channels, out_channels)
            self.a_lin[node_type] = Linear(out_channels, out_channels)
            self.skip[node_type] = Parameter(torch.Tensor(1))

        if use_edge_type_params:
            self.a_rel = torch.nn.ParameterDict()
            self.m_rel = torch.nn.ParameterDict()

            dim = out_channels // heads
            for edge_type in metadata[1]:
                edge_type = '__'.join(edge_type)
                self.a_rel[edge_type] = Parameter(torch.Tensor(heads, dim, dim))
                self.m_rel[edge_type] = Parameter(torch.Tensor(heads, dim, dim))

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.k_lin)
        reset(self.q_lin)
        reset(self.v_lin)
        reset(self.a_lin)
        ones(self.skip)

        if self.use_edge_type_params:
            glorot(self.a_rel)
            glorot(self.m_rel)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Union[Dict[EdgeType, Tensor],
                               Dict[EdgeType, SparseTensor]]
    ) -> Dict[NodeType, Optional[Tensor]]:
        H, D = self.heads, self.out_channels // self.heads

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        # Iterate over node-types:
        for node_type, x in x_dict.items():
            k_dict[node_type] = self.k_lin[node_type](x).view(-1, H, D)
            q_dict[node_type] = self.q_lin[node_type](x).view(-1, H, D)
            v_dict[node_type] = self.v_lin[node_type](x).view(-1, H, D)

            # out_dict: node type 별 벡터 결과물
            out_dict[node_type] = []

        # Iterate over edge-types:
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)

            if self.use_edge_type_params:
                a_rel = self.a_rel[edge_type]
                m_rel = self.m_rel[edge_type]

                # k_dict[src_type] = (b, heads, hidden) --transpose--> (heads, b, hidden)
                # a_rel = (heads, hidden, hidden)
                # k = (heads, b, hidden) --transpose--> (b, heads, hihdden)
                k = (k_dict[src_type].transpose(0, 1) @ a_rel).transpose(1, 0)
                v = (v_dict[src_type].transpose(0, 1) @ m_rel).transpose(1, 0)
            else:
                k = k_dict[src_type]
                v = v_dict[src_type]

            # propagate_type: (k: Tensor, q: Tensor, v: Tensor)
            out = self.propagate(
                edge_index,
                k=k, q=q_dict[dst_type], v=v,
                size=None)
            out_dict[dst_type].append(out)

        # Iterate over node-types:
        for node_type, outs in out_dict.items():
            # target-specific aggregation
            out = group(outs, self.group)

            if out is None:
                out_dict[node_type] = None
                continue

            out = F.gelu(out)
            out = self.a_lin[node_type](out)

            # residual connection
            # 이전 layer의 벡터 차원과 현재 layer의 벡터 차원이 같아야만 바로 더할 수 있음
            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = self.skip[node_type].sigmoid()
                out = alpha * out + (1 - alpha) * x_dict[node_type]
            out_dict[node_type] = out

        return out_dict

    def message(
        self,
        k_j: Tensor,
        q_i: Tensor,
        v_j: Tensor,
        index: Tensor,
        ptr: Optional[Tensor],
        size_i: Optional[int]
    ) -> Tensor:

        # attention score
        alpha = (q_i * k_j).sum(dim=-1)
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)

        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        identity = f'{self.__class__.__name__}({self.out_channels}, heads={self.heads})'
        return identity
