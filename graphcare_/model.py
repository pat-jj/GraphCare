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
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import softmax
from torch.nn import LeakyReLU


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
            freeze=False, patient_mode="joint", use_alpha=True, use_beta=True, use_edge_attn=True, 
            self_attn=0., gnn="BAT", attn_init=None, drop_rate=0.,
        ):
        super(GraphCare, self).__init__()

        self.gnn = gnn
        self.embedding_dim = embedding_dim
        self.decay_rate = decay_rate
        self.patient_mode = patient_mode
        self.use_alpha = use_alpha
        self.use_beta = use_beta
        self.edge_attn = use_edge_attn
        self.drop_rate = drop_rate
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.max_visit = max_visit

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


        for layer in range(1, layers+1):
            if self.use_alpha:
                self.alpha_attn[str(layer)] = nn.Linear(num_nodes, num_nodes)

                if attn_init is not None:
                    attn_init = attn_init.float()  # Convert attn_init to float
                    attn_init_matrix = torch.eye(num_nodes).float() * attn_init  # Multiply the identity matrix by attn_init
                    self.alpha_attn[str(layer)].weight.data.copy_(attn_init_matrix)  # Copy the modified attn_init_matrix to the weights

                else:
                    nn.init.xavier_normal_(self.alpha_attn[str(layer)].weight)
            if self.use_beta:
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

        # we found that batch normalization is not helpful
        # x = self.bn1(self.lin(x))
        # edge_attr = self.bn1(self.lin(edge_attr))

        x = self.lin(x)
        edge_attr = self.lin(edge_attr)


        if store_attn:
            self.alpha_weights = []
            self.beta_weights = []
            self.attention_weights = []
            self.edge_weights = []

        for layer in range(1, self.layers+1):
            if self.use_alpha:
                # alpha = masked_softmax((self.leakyrelu(self.alpha_attn[str(layer)](visit_node.float()))), mask=visit_node>1, dim=1)
                alpha = torch.softmax((self.alpha_attn[str(layer)](visit_node.float())), dim=1)  # (batch, max_visit, num_nodes)

            if self.use_beta:
                # beta = masked_softmax((self.leakyrelu(self.beta_attn[str(layer)](visit_node.float()))), mask=visit_node>1, dim=0) * self.lambda_j
                beta = torch.tanh((self.beta_attn[str(layer)](visit_node.float()))) * self.lambda_j

            if self.use_alpha and self.use_beta:
                attn = alpha * beta
            elif self.use_alpha:
                attn = alpha * torch.ones((batch.max().item() + 1, self.max_visit, 1)).to(edge_index.device)
            elif self.use_beta:
                attn = beta * torch.ones((batch.max().item() + 1, self.max_visit, self.num_nodes)).to(edge_index.device)
            else:
                attn = torch.ones((batch.max().item() + 1, self.max_visit, self.num_nodes)).to(edge_index.device)
                
            attn = torch.sum(attn, dim=1)
            
            xj_node_ids = node_ids[edge_index[0]]
            xj_batch = batch[edge_index[0]]
            attn = attn[xj_batch, xj_node_ids].reshape(-1, 1)

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


