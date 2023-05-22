import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GATConv, GINConv
from pyhealth.models import RETAINLayer
from torch_geometric.nn.inits import reset

from typing import Callable, Optional, Union

import torch
import random
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from torch_geometric.utils import spmm
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import softmax
from torch.nn import LeakyReLU

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class BiAttentionGNNConv(MessagePassing):
    def __init__(self, nn: torch.nn.Module, eps: float = 0.,
                 train_eps: bool = False, edge_dim: Optional[int] = None,
                 edge_attn=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        self.edge_attn = edge_attn
        if edge_attn:
            # self.W_R = torch.nn.Linear(edge_dim, edge_dim)
            self.W_R = torch.nn.Linear(edge_dim, 1)
        else:
            self.W_R = None

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        self.reset_parameters()

    def reset_parameters(self):
        self.nn.reset_parameters()
        self.eps.data.fill_(self.initial_eps)
        if self.W_R is not None:
            self.W_R.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None, attn: Tensor = None) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size, attn=attn)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        if self.W_R is not None:
            w_rel = self.W_R(edge_attr)
        else:
            w_rel = None

        return self.nn(out), w_rel

    def message(self, x_j: Tensor, edge_attr: Tensor, attn: Tensor) -> Tensor:

        if self.edge_attn:
            w_rel = self.W_R(edge_attr)
            out = (x_j * attn + w_rel * edge_attr).relu()
        else:
            out = (x_j * attn).relu()
        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


def masked_softmax(src: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    out = src.masked_fill(~mask, float('-inf'))
    out = torch.softmax(out, dim=dim)
    out = out.masked_fill(~mask, 0)
    return out + 1e-8  # Add small constant to avoid numerical issues

class GraphCare(nn.Module):
    def __init__(
            self, num_nodes, num_rels, max_visit, embedding_dim, hidden_dim, 
            out_channels, layers=3, dropout=0.5, decay_rate=0.03, node_emb=None, rel_emb=None,
            freeze=False, patient_mode="joint", use_alpha=True, use_beta=True, use_gamma=False, use_edge_attn=True, 
            self_attn=0., gnn="BAT", attn_init=None, drop_rate=0.,
        ):
        super(GraphCare, self).__init__()

        self.gnn = gnn
        self.embedding_dim = embedding_dim
        self.decay_rate = decay_rate
        self.patient_mode = patient_mode
        self.use_alpha = use_alpha
        self.use_beta = use_beta
        self.use_gamma = use_gamma
        self.edge_attn = use_edge_attn
        self.drop_rate = drop_rate
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.max_visit = max_visit
        self.batch_size = 4

        j = torch.arange(max_visit).float()
        self.lambda_j = torch.exp(self.decay_rate * (max_visit - j)).unsqueeze(0).reshape(1, max_visit, 1).float()

        if node_emb is None:
            self.node_emb = nn.Embedding(num_nodes, embedding_dim)
        else:
            self.node_emb = nn.Embedding.from_pretrained(node_emb, freeze=freeze)

        if rel_emb is None:
            self.rel_emb = nn.Embedding(num_rels, embedding_dim)
        else:
            self.rel_emb = nn.Embedding.from_pretrained(rel_emb, freeze=freeze)

        self.lin = nn.Linear(embedding_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.layers = layers
        self.dropout = dropout

        self.alpha_attn = nn.ModuleDict()
        self.beta_attn = nn.ModuleDict()
        self.conv = nn.ModuleDict()
        self.bn_gnn = nn.ModuleDict()

        self.leakyrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.tahh = nn.Tanh()

        if attn_init is not None:
            self.attn_init = attn_init.unsqueeze(0).unsqueeze(0).expand((self.batch_size, max_visit, num_nodes, 1)).squeeze(-1).requires_grad_(True)
        else:
            self.attn_init = None


        for layer in range(1, layers+1):
            if self.use_alpha:
                self.alpha_attn[str(layer)] = nn.Linear(hidden_dim, 1)
                nn.init.xavier_normal_(self.alpha_attn[str(layer)].weight)
            if self.use_beta:
                # self.beta_attn[str(layer)] = nn.Linear(hidden_dim, 1)
                self.beta_attn[str(layer)] = nn.Linear(num_nodes, 1)
                nn.init.xavier_normal_(self.beta_attn[str(layer)].weight)

            if self.gnn == "BAT":
                self.conv[str(layer)] = BiAttentionGNNConv(nn.Linear(hidden_dim, hidden_dim), edge_dim=hidden_dim, edge_attn=self.edge_attn, eps=self_attn)
            elif self.gnn == "GAT":
                self.conv[str(layer)] = GATConv(hidden_dim, hidden_dim)
            elif self.gnn == "GIN":
                self.conv[str(layer)] = GINConv(nn.Linear(hidden_dim, hidden_dim))

            # self.bn_gnn[str(layer)] = nn.BatchNorm1d(hidden_dim)


        if self.patient_mode == "joint":
            self.MLP = nn.Linear(hidden_dim * 2, out_channels)
        else:
            self.MLP = nn.Linear(hidden_dim, out_channels)


    def to(self, device):
        super().to(device)
        self.lambda_j = self.lambda_j.float().to(device)
        if self.attn_init is not None:
            self.attn_init = self.attn_init.float().to(device)
        return self


    def forward(self, node_ids, rel_ids, edge_index, batch, visit_node, ehr_nodes, store_attn=False, in_drop=False):
        
        if in_drop and self.drop_rate > 0:
            edge_count = edge_index.size(1)
            edges_to_remove = int(edge_count * self.drop_rate)
            indices_to_remove = set(random.sample(range(edge_count), edges_to_remove))
            edge_index = edge_index[:, [i for i in range(edge_count) if i not in indices_to_remove]].to(edge_index.device)
            rel_ids = torch.tensor([rel_id for i, rel_id in enumerate(rel_ids) if i not in indices_to_remove], device=rel_ids.device)

        x = self.node_emb(node_ids).float()
        edge_attr = self.rel_emb(rel_ids).float()

        x = self.lin(x)
        edge_attr = self.lin(edge_attr)

        # start of new code
        max_nodes = self.num_nodes  # Maximum number of nodes in any sample
        batch_size = batch.max().item() + 1  # Number of samples in the batch
        max_visits = visit_node.shape[1]  # Maximum number of visits in any sample
        hidden_dim = x.shape[1]  # Hidden dimension size

        x_new = torch.zeros((batch_size, max_visits, max_nodes, hidden_dim), device=x.device)

        for batch_id in range(batch_size):
            x_graph = x[batch == batch_id]
            num_nodes = x_graph.shape[0]
            x_new[batch_id, :, :num_nodes] = x_graph

        x_attn = x_new # patients, visits, nodes, features

        if store_attn:
            self.alpha_weights = []
            self.beta_weights = []
            self.attention_weights = []
            self.edge_weights = []

        for layer in range(1, self.layers+1):
            if self.use_alpha:
                if layer == 1:
                    visit_node_ = visit_node.unsqueeze(-1) # patients, visits, nodes, 1
                feature_mat = visit_node_.float() * x_attn # patients, visits, nodes, features

                alpha_lin = self.alpha_attn[str(layer)](feature_mat).squeeze(-1) # patients, visits, nodes
                
                if self.attn_init is not None:
                    # attn_init # nodes, 1
                    # alpha_lin += self.attn_init.unsqueeze(0).unsqueeze(0).expand(alpha_lin.shape).squeeze(-1) # patients, visits, nodes
                    alpha_lin * self.attn_init # patients, visits, nodes
                alpha = torch.softmax(alpha_lin, dim=1) # patients, visits, nodes

            if self.use_beta:
                beta = torch.tanh((self.beta_attn[str(layer)](visit_node.float()))) * self.lambda_j # patients, visits, nodes, 1

                # 2
                # feature_mat = visit_node.float() * x_attn # patients, visits, nodes, features
                # visit_emb = torch.mean(feature_mat, dim=2) # patients, visits, features
                # beta_lin = self.beta_attn[str(layer)](visit_emb) # patients, visits, 1
                # beta = torch.tanh(beta_lin) * self.lambda_j # patients, visits, 1

                # 3
                # beta = torch.softmax(beta_lin, dim=0) * self.lambda_j # patients, visits, 1
                # beta = beta.unsqueeze(2)  # Add a dimension for nodes
                # beta = beta.expand(-1, -1, alpha.shape[2], -1).squeeze(-1)  # patients, visits, nodes

            if self.use_alpha and self.use_beta:
                attn = alpha * beta  # patients, visits, nodes

            elif self.use_alpha:
                attn = alpha.squeeze(-1)  # patients, visits, nodes
            elif self.use_beta:
                attn = beta.squeeze(-1)  # patients, visits, nodes
            else:
                attn = None
            
            xj_node_ids = node_ids[edge_index[0]]
            xj_batch = batch[edge_index[0]]
            xj_visit_ids = torch.zeros(len(edge_index[0])).long().to(edge_index.device)
            attn = attn[xj_batch, xj_visit_ids, xj_node_ids].reshape(-1, 1) # edges, 1


            if self.gnn == "BAT":
                x, w_rel = self.conv[str(layer)](x, edge_index, edge_attr, attn=attn)

            else:
                x = self.conv[str(layer)](x, edge_index)
            
            # x = self.bn_gnn[str(layer)](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

            if store_attn:
                self.alpha_weights.append(alpha)
                self.beta_weights.append(beta)
                self.attention_weights.append(attn)
                self.edge_weights.append(w_rel)

        if self.patient_mode == "joint" or self.patient_mode == "graph":
            # patient graph embedding through global mean pooling
            # x = x.mean(dim=1)  # patients, nodes, features
            # x_graph = x.mean(dim=1)  # patients, features
            x_graph = global_mean_pool(x, batch)
            x_graph = F.dropout(x_graph, p=self.dropout, training=self.training)


        if self.patient_mode == "joint" or self.patient_mode == "node":
            # patient node embedding through local (direct EHR) mean pooling
            x_node = torch.stack([ehr_nodes[i].view(1, -1) @ self.node_emb.weight / torch.sum(ehr_nodes[i]) for i in range(batch.max().item() + 1)])
            x_node = self.lin(x_node).squeeze(1)
            x_node = F.dropout(x_node, p=self.dropout, training=self.training)

        if self.patient_mode == "joint":
            # concatenate patient graph embedding and patient node embedding
            x_concat = torch.cat((x_graph, x_node), dim=1)
            x_concat = F.dropout(x_concat, p=self.dropout, training=self.training)
            # MLP for prediction
            logits = self.MLP(x_concat)

        elif self.patient_mode == "graph":
            # MLP for prediction
            logits = self.MLP(x_graph)
        
        elif self.patient_mode == "node":
            # MLP for prediction
            logits = self.MLP(x_node)

        if store_attn:
            return logits, self.alpha_weights, self.beta_weights,  self.attention_weights, self.edge_weights
        else:
            return logits


