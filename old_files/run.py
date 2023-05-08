import json

file_dir = "./graphs/condition/CCSCM"

file_id2ent = f"{file_dir}/id2ent.json"
file_ent2id = f"{file_dir}/ent2id.json"
file_id2rel = f"{file_dir}/id2rel.json"
file_rel2id = f"{file_dir}/rel2id.json"

with open(file_id2ent, 'r') as file:
    cond_id2ent = json.load(file)
with open(file_ent2id, 'r') as file:
    cond_ent2id = json.load(file)
with open(file_id2rel, 'r') as file:
    cond_id2rel = json.load(file)
with open(file_rel2id, 'r') as file:
    cond_rel2id = json.load(file)


import csv

condition_mapping_file = "./resources/CCSCM.csv"
procedure_mapping_file = "./resources/CCSPROC.csv"
drug_file = "./resources/ATC.csv"

condition_dict = {}
with open(condition_mapping_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        condition_dict[row['code']] = row['name'].lower()

procedure_dict = {}
with open(procedure_mapping_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        procedure_dict[row['code']] = row['name'].lower()

drug_dict = {}
with open(drug_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['level'] == '5.0':
            drug_dict[row['code']] = row['name'].lower()


def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


from pyhealth.tokenizer import Tokenizer
import numpy as np
from tqdm import tqdm
import torch

def multihot(label, num_labels):
    multihot = np.zeros(num_labels)
    for l in label:
        multihot[l] = 1
    return multihot

def prepare_label(drugs):
    label_tokenizer = Tokenizer(
        sample_dataset.get_all_tokens(key='drugs')
    )

    labels_index = label_tokenizer.convert_tokens_to_indices(drugs)
    # print(labels_index)
    # convert to multihot
    num_labels = label_tokenizer.get_vocabulary_size()
    # print(num_labels)
    labels = multihot(labels_index, num_labels)
    return labels


# for patient in tqdm(sample_dataset):
#     # patient['drugs_all'] = flatten(patient['drugs'])
#     # print(patient['drugs_all'])
#     patient['drugs_ind'] = torch.tensor(prepare_label(patient['drugs']))

import pickle

with open('./exp_data/ccscm_ccsproc/sample_dataset.pkl', 'rb') as f:
    sample_dataset= pickle.load(f)


from pyhealth.datasets import split_by_patient

train_dataset, val_dataset, test_dataset = split_by_patient(sample_dataset, [0.8, 0.1, 0.1], seed=528)

from tqdm import tqdm
import numpy as np
import networkx as nx
import pickle

with open('./graphs/cond_proc/CCSCM_CCSPROC/ent2id.json', 'r') as file:
    ent2id = json.load(file)
with open('./graphs/cond_proc/CCSCM_CCSPROC/rel2id.json', 'r') as file:
    rel2id = json.load(file)
with open('./graphs/cond_proc/CCSCM_CCSPROC/entity_embedding.pkl', 'rb') as file:
    ent_emb = pickle.load(file)
    

G = nx.Graph()

for i in range(len(ent_emb)):
    G.add_nodes_from([
        (i, {'y': i, 'x': ent_emb[i]})
    ])

triples_all = []
for patient in tqdm(sample_dataset):
    triples = []
    triple_set = set()
    # node_set = set()
    conditions = flatten(patient['conditions'])
    for condition in conditions:
        cond_file = f'./graphs/condition/CCSCM/{condition}.txt'
        with open(cond_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            items = line.split('\t')
            if len(items) == 3:
                h, r, t = items
                t = t[:-1]
                h = int(ent2id[h])
                r = int(rel2id[r])
                t = int(ent2id[t])
                triple = (h, r, t)
                if triple not in triple_set:
                    triples.append((h, t))
                    triple_set.add(triple)
                    # node_set.add(h)
                    # node_set.add(r)
    procedures = flatten(patient['procedures'])
    for procedure in procedures:
        proc_file = f'./graphs/procedure/CCSPROC/{procedure}.txt'
        with open(proc_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            items = line.split('\t')
            if len(items) == 3:
                h, r, t = items
                t = t[:-1]
                h = int(ent2id[h])
                r = int(rel2id[r])
                t = int(ent2id[t])
                triple = (h, r, t)
                if triple not in triple_set:
                    triples.append((h, t))
                    triple_set.add(triple)

    G.add_edges_from(
        triples,
        # label=prepare_label(patient['drugs'])
    )
    
    # triples.append(prepare_label(patient['drugs']))
    # triples_all.append(np.array(triples))

from torch_geometric.utils import to_networkx, from_networkx
import pickle

G_tg = from_networkx(G)

with open('./exp_data/ccscm_ccsproc/graph_tg.pkl', 'wb') as f:
    pickle.dump(G, f)


import torch

def get_subgraph(dataset, idx):
    
    subgraph_list = []
    # for patient in tqdm(dataset):
    patient = dataset[idx]
    triple_set = set()
    node_set = set()
    conditions = flatten(patient['conditions'])
    for condition in conditions:
        cond_file = f'./graphs/condition/CCSCM/{condition}.txt'
        with open(cond_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            items = line.split('\t')
            if len(items) == 3:
                h, r, t = items
                t = t[:-1]
                h = int(ent2id[h])
                r = int(rel2id[r])
                t = int(ent2id[t])
                triple = (h, r, t)
                if triple not in triple_set:
                    triple_set.add(triple)
                    node_set.add(h)
                    node_set.add(r)

    procedures = flatten(patient['procedures'])
    for procedure in procedures:
        proc_file = f'./graphs/procedure/CCSPROC/{procedure}.txt'
        with open(proc_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            items = line.split('\t')
            if len(items) == 3:
                h, r, t = items
                t = t[:-1]
                h = int(ent2id[h])
                r = int(rel2id[r])
                t = int(ent2id[t])
                triple = (h, r, t)
                if triple not in triple_set:
                    triple_set.add(triple)
                    node_set.add(h)
                    node_set.add(r)

    P = G_tg.subgraph(torch.tensor([*node_set]))
    P.label = patient['drugs_ind']
        # subgraph_list.append(P)

    return P


from torch_geometric.loader import DataListLoader, DataLoader

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset=dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return get_subgraph(dataset=self.dataset, idx=idx)

train_set = Dataset(train_dataset)
val_set = Dataset(val_dataset)
test_set = Dataset(test_dataset)


import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import DataParallel
from torch_geometric.loader import DataListLoader
import pickle

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels*heads, hidden_channels, heads=heads)
        self.conv3 = GATConv(hidden_channels*heads, hidden_channels, heads=1)

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


from torch_geometric.nn import GINConv

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


from torch_geometric.nn import HGTConv

class HGT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2):
        super(HGT, self).__init__()
        self.conv1 = HGTConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = HGTConv(hidden_channels*heads, hidden_channels, heads=heads)
        self.conv3 = HGTConv(hidden_channels*heads, hidden_channels, heads=1)

        self.fc = Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        logits = self.fc(x)
        return logits


from tqdm import tqdm

def train(model, device, train_loader, optimizer):
    model.train()
    training_loss = 0
    pbar = tqdm(train_loader)
    for data in pbar:
        pbar.set_description(f'loss: {training_loss}')
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        try:
            label = data.label.reshape(int(train_loader.batch_size), int(len(data.label)/train_loader.batch_size))
        except:
            continue
        loss = F.binary_cross_entropy_with_logits(out, label)
        loss.backward()
        training_loss = loss
        optimizer.step()
    
    return training_loss 


from pyhealth.metrics import multilabel_metrics_fn

def evaluate(model, device, loader):
    model.eval()
    y_prob_all = []
    y_true_all = []

    for data in tqdm(loader):
        data = data.to(device)
        with torch.no_grad():
            logits = model(data.x, data.edge_index, data.batch)
            y_prob = torch.sigmoid(logits)
            try:
                y_true = data.label.reshape(int(loader.batch_size), int(len(data.label)/loader.batch_size))
            except:
                continue
            y_prob_all.append(y_prob.cpu())
            y_true_all.append(y_true.cpu())
            
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)
    # pr_auc = multilabel_metrics_fn(y_true=y_true_all, y_prob=y_true_all, metrics="pr_auc_macro")

    return y_true_all, y_prob_all


from sklearn.metrics import average_precision_score
import torch.distributed as dist

def train_loop(train_loader, val_loader, model, optimizer, device, epochs):
    for epoch in range(1, epochs+1):
        loss = train(model, device, train_loader, optimizer)
        dist.barrier()
        # y_true_all, y_prob_all = evaluate(model, device, train_loader)
        # train_pr_auc = average_precision_score(y_true_all, y_prob_all, average="macro")
        y_true_all, y_prob_all = evaluate(model, device, val_loader)
        val_pr_auc = average_precision_score(y_true_all, y_prob_all, average="samples")
        print(f'Epoch: {epoch}, Training loss: {loss}, Val PRAUC: {val_pr_auc:.4f}')
        dist.barrier()


import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

def run(rank, world_size, root):
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                       rank=rank)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, sampler=train_sampler)

    in_channels = train_set[0].x.shape[1]
    out_channels = len(train_set[0].label)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAT(in_channels=in_channels, out_channels=out_channels, hidden_channels=512, heads=2).to(device)
    # model = GIN(in_channels=in_channels, out_channels=out_channels, hidden_channels=512).to(device)
    # model = HGT(in_channels=in_channels, out_channels=out_channels, hidden_channels=512, heads=2).to(device)
    model.double()
    model = DistributedDataParallel(model, device_ids=[rank])

    if rank == 0:
        val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loop(train_loader=train_loader, val_loader=val_loader, model=model, optimizer=optimizer, device=device, epochs=100)


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    root = "./exp_data"
    args=(world_size, root)
    mp.spawn(run, args=args, nprocs=world_size, join=True)