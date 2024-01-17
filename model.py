import torch
import torch.nn as nn
#from torch.nn import Sequential, Linear, ReLU, LeakyReLU, BatchNorm1d
from torch_geometric.utils import scatter
from torch_geometric.nn import MetaLayer

class EdgeModel(torch.nn.Module):
    def __init__(self, node_in, edge_in, edge_out, leakiness=0.0):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.BatchNorm1d(2*node_in + edge_in), ##2 x nodes (one for each node in pair) + edges
            nn.Linear(2*node_in + edge_in, edge_out),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.BatchNorm1d(edge_out),
            nn.Linear(edge_out, edge_out),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.BatchNorm1d(edge_out),
            nn.Linear(edge_out, edge_out)
        )

    def forward(self, src, dst, edge_attr, u=None, batch=None):
        out = torch.cat([src, dst, edge_attr], dim=1)
        return self.edge_mlp(out)

class NodeModel(torch.nn.Module):
    def __init__(self, node_in, node_out, edge_in, leakiness=0.0):
        super().__init__()
        self.node_mlp_1 = nn.Sequential(
            nn.BatchNorm1d(node_in + edge_in),
            nn.Linear(node_in + edge_in, node_out),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.BatchNorm1d(node_out),
            nn.Linear(node_out, node_out),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.BatchNorm1d(node_out),
            nn.Linear(node_out, node_out)
        )

        self.node_mlp_2 = nn.Sequential(
            nn.BatchNorm1d(node_in + node_out),
            nn.Linear(node_in + node_out, node_out),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.BatchNorm1d(node_out),
            nn.Linear(node_out, node_out),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.BatchNorm1d(node_out),
            nn.Linear(node_out, node_out)
        )

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter(out, col, dim=0, dim_size=x.size(0),
                      reduce='mean')
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)

class GlobalModel(torch.nn.Module):
    def __init__(self, node_in, node_out, leakiness=0.0):
        super().__init__()
        self.global_mlp = nn.Sequential(
            nn.BatchNorm1d(node_in),
            nn.Linear(node_in, node_out),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.BatchNorm1d(node_out),
            nn.Linear(node_out, node_out),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.BatchNorm1d(node_out),
            nn.Linear(node_out, node_out)
        )

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        #out = torch.cat([
        #    scatter(x, batch, dim=0, reduce='mean'),
        #], dim=1)
        out = scatter(x, batch, dim=0, reduce='mean')
        return self.global_mlp(out)

class GNNModel(nn.Module):
    def __init__(
        self,
        do_frac=False,
        outdim=6,
        node_input=9,
        edge_input=12,
        node_output=64,
        edge_output=64,
    ):
        super().__init__()

        # Initialize the updaters
        #self.message_passing = torch.nn.ModuleList()

        # Update the node and edge feature N times (number of message passings, here = 3)
        #node_input  = 9 # Number of input node features
        #edge_input  = 12 # Number of input edge features -- update when ready
        #node_output = 64  # Number of intermediate node features
        #edge_output = 64  # Number of intermediate edge features
        leakiness = 0.1 # LeakyRELU activation leakiness
        #self.num_mp = 3 # Number of message passings

        self.message_pass = MetaLayer(
          edge_model=EdgeModel(node_input, edge_input, edge_output), #Add leakiness?
          node_model=NodeModel(node_input, node_output, edge_output), #Add leakiness?
          global_model=GlobalModel(node_output, node_output), #Add leakiness?
        )
        self.outdim = outdim
        #for i in range(self.num_mp):
        #    self.message_passing.append(
        #        MetaLayer(
        #            edge_model = EdgeLayer(node_input, edge_input, edge_output, leakiness=leakiness),
        #            node_model = NodeLayer(node_input, node_output, edge_output, leakiness=leakiness)
        #        )
        #    )
        #    node_input = node_output
        #    edge_input = edge_output
        #    global_input = globa_output


        # Reduce the number of node and edge features edge, as we are performing a simple classification
        #self.node_predictor = nn.Linear(node_output, 2)
        #self.edge_predictor = nn.Linear(edge_output, 2)
        if do_frac:
          self.global_predictor = nn.Sequential(
            nn.Linear(node_output, 1),
            nn.Sigmoid()
          )
        else:
          self.global_predictor = nn.Sequential(
            nn.Linear(node_output, self.outdim),
            nn.Softmax(dim=1)
          )

    def forward(self, data, batch):

        # Loop over message passing steps, pass data through the updaters
        x = data.x
        e = data.edge_attr
        x, e, u = self.message_pass(x, data.edge_index, e, u=None, batch=batch)
        #u = data.global_attr
        #for i in range(self.num_mp):
        #    x, e, u = self.message_passing[i](x, data.edge_index, e, u, batch=data.batch)

        # Reduce output features to 2 each
        #x_pred = self.node_predictor(x)
        #e_pred = self.edge_predictor(e)
        u_pred = self.global_predictor(u)
        return u_pred

class NuModel(nn.Module):
    def __init__(
        self,
        active_branches={
          'sign':1,
          'flavor':4,
          'mode':4,
          'protons':4,
          'pions':4,
          'pi0s':4,
          'neutrons':4,
        },
        node_input=9,
        edge_input=12,
        node_output=64,
        edge_output=64,
    ):
        super().__init__()

        # Initialize the updaters
        #self.message_passing = torch.nn.ModuleList()

        # Update the node and edge feature N times (number of message passings, here = 3)
        leakiness = 0.1 # LeakyRELU activation leakiness
        #self.num_mp = 3 # Number of message passings

        self.message_pass = MetaLayer(
          edge_model=EdgeModel(node_input, edge_input, edge_output), #Add leakiness?
          node_model=NodeModel(node_input, node_output, edge_output), #Add leakiness?
          global_model=GlobalModel(node_output, node_output), #Add leakiness?
        )
        self.branches = active_branches
        #for i in range(self.num_mp):
        #    self.message_passing.append(
        #        MetaLayer(
        #            edge_model = EdgeLayer(node_input, edge_input, edge_output, leakiness=leakiness),
        #            node_model = NodeLayer(node_input, node_output, edge_output, leakiness=leakiness)
        #        )
        #    )
        #    node_input = node_output
        #    edge_input = edge_output
        #    global_input = globa_output


        # Reduce the number of node and edge features edge, as we are performing a simple classification
        #self.node_predictor = nn.Linear(node_output, 2)
        #self.edge_predictor = nn.Linear(edge_output, 2)

        #self.global_predictor = nn.Sequential(
        #  nn.Linear(node_output, self.outdim),
        #  nn.Softmax(dim=1)
        #)

        #Make the dict of global predictors for each active branch
        self.global_predictors = {
          name:nn.Sequential(
            nn.Linear(node_output, outdim),
            nn.Sigmoid() if outdim==1 else nn.Softmax(dim=1)
          ) for name,outdim in self.branches.items()
        }

    def forward(self, data, batch):

        # Loop over message passing steps, pass data through the updaters
        x = data.x
        e = data.edge_attr
        x, e, u = self.message_pass(x, data.edge_index, e, u=None, batch=batch)

        u_pred = {
          name:pred(u) for name,pred in self.global_predictors.items()
        }
        return u_pred

