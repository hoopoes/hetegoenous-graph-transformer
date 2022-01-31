import os
import gc

import torch
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.datasets import DBLP
import torch_geometric.transforms as T
from torch_geometric.nn import Linear

from models import HGTConv
from utils import get_device


def get_hetero_data():
    path = os.path.join(os.getcwd(), 'data/DBLP')
    dataset = DBLP(path)
    data = dataset[0]

    hetero_data = HeteroData()

    hetero_data['author'].x = data['author'].x
    hetero_data['author'].y = data['author'].y

    # torch.full((data['author'].x.shape[0], ), True)
    hetero_data['author'].train_mask = data['author']['train_mask']
    hetero_data['author'].val_mask = data['author']['val_mask']
    hetero_data['author'].test_mask = data['author']['test_mask']

    hetero_data['paper'].x = data['paper'].x
    hetero_data['paper'].train_mask = torch.full((data['paper'].x.shape[0], ), True)

    hetero_data['paper', 'to', 'author'].edge_index = data['paper', 'to', 'author']['edge_index']
    hetero_data = T.ToUndirected()(hetero_data)

    return hetero_data


hetero_data = get_hetero_data()


class HGT(torch.nn.Module):
    def __init__(
            self,
            hetero_data,
            hidden_channels,
            out_channels,
            num_heads,
            num_layers
    ):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in hetero_data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                metadata=hetero_data.metadata(),
                heads=num_heads,
                group='sum')
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        output = x_dict
        return output


device = get_device()
model = HGT(hetero_data=hetero_data, hidden_channels=64, out_channels=4, num_heads=2, num_layers=1)
device = torch.device(device)
hetero_data, model = hetero_data.to(device), model.to(device)

# Initialize lazy modules
with torch.no_grad():
    out = model(hetero_data.x_dict, hetero_data.edge_index_dict)

print(f"initialized HGT model with {sum([p.numel() for p in model.parameters()])} params")

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)


def train():
    model.train()
    optimizer.zero_grad()

    out = model(hetero_data.x_dict, hetero_data.edge_index_dict)
    out = out['author']

    mask = hetero_data['author'].train_mask

    loss = F.cross_entropy(out[mask], hetero_data['author'].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(hetero_data.x_dict, hetero_data.edge_index_dict)
    pred = pred['author']
    pred = pred.argmax(dim=-1)

    accs = []
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = hetero_data['author'][split]
        acc = (pred[mask] == hetero_data['author'].y[mask]).sum() / mask.sum()
        accs.append(float(acc))
    return accs


for epoch in range(1, 101):
    loss = train()
    train_acc, val_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')


params = {n: p for n, p in model.named_parameters()}
print(params['convs.0.skip.author'].detach().cpu().numpy()[0],
      params['convs.0.skip.paper'].detach().cpu().numpy()[0])

