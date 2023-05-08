import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# Define the GAT model
class GAT(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, num_drugs):
        super(GAT, self).__init__()
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
        self.conv1 = GATConv(embedding_dim, embedding_dim, heads=8, dropout=0.6)
        self.conv2 = GATConv(8*embedding_dim, embedding_dim, dropout=0.6)
        self.fc = nn.Linear(embedding_dim, num_drugs)

    def forward(self, x, edge_index, edge_type):
        entity_embed = self.entity_embedding(x)
        relation_embed = self.relation_embedding(edge_type)
        x = torch.cat([entity_embed, relation_embed], dim=-1)
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = torch.cat([x, entity_embed], dim=-1)
        x = F.elu(self.conv2(x, edge_index))
        x = torch.mean(x, dim=0, keepdim=True)
        x = self.fc(x)
        return x

