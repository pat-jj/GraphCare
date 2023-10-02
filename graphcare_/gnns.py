import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv, GINConv, HGTConv
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import DataParallel
from torch_geometric.loader import DataListLoader


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(GAT, self).__init__()
        
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels*heads, hidden_channels, heads=heads)
        self.conv3 = GATConv(hidden_channels*heads, hidden_channels, heads=heads)
        self.conv4 = GATConv(hidden_channels*heads, hidden_channels, heads=heads)

        self.fc = Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, batch):
        
        x = F.elu(self.conv1(x, edge_index))
        # print(x.shape)
        x = F.elu(self.conv2(x, edge_index))
        # print(x.shape)
        x = F.elu(self.conv3(x, edge_index))
        # print(x.shape)
        x = global_mean_pool(x, batch)
        # print(x.shape)
        x = F.dropout(x, p=0.5, training=self.training)
        # print(x.shape)
        logits = self.fc(x)
        # print(logits.shape)
        return logits


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GIN, self).__init__()
        self.conv1 = GINConv(Linear(in_channels, hidden_channels))
        self.conv2 = GINConv(Linear(hidden_channels, hidden_channels))
        self.conv3 = GINConv(Linear(hidden_channels, hidden_channels))

        self.fc = Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        logits = self.fc(x)
        return logits