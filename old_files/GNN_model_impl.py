import pickle
import json
from pyhealth.datasets import SampleDataset
from pyhealth.datasets import split_by_patient
from torch_geometric.utils import to_networkx, from_networkx
import torch
from torch_geometric.loader import DataListLoader, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from pyhealth.models import RETAINLayer
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv, GINConv, HGTConv
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import DataParallel
from torch_geometric.loader import DataListLoader
from tqdm import tqdm
from pyhealth.metrics import multilabel_metrics_fn
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, jaccard_score


mode = "node"
kg = "GPT-KG"
# mode = "graph"
# kg = "UMLS-KG"

if kg == "UMLS-KG":
    with open('../../../data/pj20/exp_data/icd9cm_icd9proc/drugrec_dataset_umls_1000.pkl', 'rb') as f:
        sample_dataset = pickle.load(f)

    with open('../../../data/pj20/exp_data/icd9cm_icd9proc/graph_umls_1000_cp.pkl', 'rb') as f:
        G = pickle.load(f)

else:
    with open('../../../data/pj20/exp_data/ccscm_ccsproc/sample_dataset_drugrec_th015.pkl', 'rb') as f:
        sample_dataset = pickle.load(f)

    with open('../../../data/pj20/exp_data/ccscm_ccsproc/graph_pd_th015.pkl', 'rb') as f:
        G = pickle.load(f)

with open('../../../data/pj20/exp_data/ccscm_ccsproc/clusters_inv_th015.json', 'r', encoding='utf-8') as f:
    map_cluster_inv = json.load(f)

with open('../../../data/pj20/exp_data/ccscm_ccsproc/clusters_th015.json', 'r', encoding='utf-8') as f:
    map_cluster = json.load(f)

with open('./graphs/cond_proc/CCSCM_CCSPROC/ent2id.json', 'r') as file:
    ent2id = json.load(file)

with open('./graphs/cond_proc/CCSCM_CCSPROC/entity_embedding.pkl', 'rb') as file:
    ent_emb = pickle.load(file)

ccscm_id2name = {}
with open('./resources/CCSCM.csv', 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        line = line.strip().split(',')
        ccscm_id2name[line[0]] = line[1]

ccsproc_id2name = {}
with open('./resources/CCSPROC.csv', 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        line = line.strip().split(',')
        ccsproc_id2name[line[0]] = line[1]


c_v, p_v, d_v = [], [], []

patient_id_set = set()

for patient in sample_dataset:
    c_v.append(len(patient['conditions']))
    p_v.append(len(patient['procedures']))
    patient_id_set.add(patient['patient_id'])

i = 0
pid_map = {}
for patient_id in patient_id_set:
    pid_map[patient_id] = i
    i += 1

for patient in sample_dataset:
    patient['patient_id'] = pid_map[patient['patient_id']]

print(max(c_v), max(p_v))
max_visits = max(c_v)

from tqdm import tqdm
import numpy as np

def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

G_new = G
ex_pat_node_num = len(G_new.nodes)
if mode == "node":
    patient_set = set()
    i = 0

    for patient in tqdm(sample_dataset):
        pat_node_id = int(ex_pat_node_num + i)
        triples = []
        nodes = []
        patient['pat_node_id'] = pat_node_id
        patient['node_set'].append(pat_node_id)
        i += 1

        # if patient['patient_id'] in patient_set:
        #     continue
        patient_set.add(patient['patient_id'])
        
        for condition in flatten(patient['conditions']):
            try:
                ehr_node = map_cluster_inv[ent2id[ccscm_id2name[condition].lower()]]
                triples.append((pat_node_id, int(ehr_node)))
                nodes.append(int(ehr_node))
            except:
                continue

        for procedure in flatten(patient['procedures']):
            try:
                ehr_node = map_cluster_inv[ent2id[ccsproc_id2name[procedure].lower()]]
                triples.append((pat_node_id, int(ehr_node)))
                nodes.append(int(ehr_node))
            except:
                continue

        nodes = np.array(nodes)

        try:
            G_new.add_nodes_from([
                (pat_node_id, {'y': pat_node_id, 'x': np.mean(ent_emb[nodes], axis=0)})
                ])
        except:
            #randomly initialize
            G_new.add_nodes_from([
                (pat_node_id, {'y': pat_node_id, 'x': np.random.rand(1536)})
                ])
        
        
        G_new.add_edges_from(triples)
        G = G_new

def get_subgraph(G, dataset, idx):
    patient = dataset[idx]
    while len(patient['node_set']) == 0:
        idx -= 1
        patient = dataset[idx]
    # L = G.edge_subgraph(torch.tensor([*patient['node_set']]))
    P = G.subgraph(torch.tensor([*patient['node_set']]))
    P.label = patient['drugs_ind']
    P.visits_cond = patient['visit_node_set_condition']
    P.visits_proc = patient['visit_node_set_procedure']
    P.patient_id = patient['pat_node_id']
    
    return P

class Dataset(torch.utils.data.Dataset):
    def __init__(self, G, dataset):
        self.G = G
        self.dataset=dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return get_subgraph(G=self.G, dataset=self.dataset, idx=idx)


G_tg = from_networkx(G) 
train_dataset, val_dataset, test_dataset = split_by_patient(sample_dataset, [0.8, 0.1, 0.1], seed=528)
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv, GINConv, HGTConv
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import DataParallel
from torch_geometric.loader import DataListLoader

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


class GINX(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden_channels, out_channels, word_emb=None):
        super(GINX, self).__init__()
        
        if word_emb == None:
            self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)
            self.conv1 = GINConv(Linear(embedding_dim, hidden_channels))
        else:
            self.embedding = torch.nn.Embedding.from_pretrained(word_emb, freeze=False)
            self.conv1 = GINConv(Linear(word_emb.shape[1], hidden_channels))

        self.conv2 = GINConv(Linear(hidden_channels, hidden_channels))
        self.conv3 = GINConv(Linear(hidden_channels, hidden_channels))
        self.fc = Linear(hidden_channels, out_channels)
        
    def forward(self, node_ids, edge_index, batch):
        x = self.embedding(node_ids)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        logits = self.fc(x)
        return logits


class GraphCare(nn.Module):
    def __init__(self, num_nodes, feature_keys, embedding_dim, hidden_dim, out_channels, dropout=0.5, max_visits=None, word_emb=None, use_attn=True, mode='graph', num_patient=None):
        super(GraphCare, self).__init__()
        self.max_visits = max_visits
        self.max_nodes = len(word_emb)
        self.embedding_dim = embedding_dim
        self.use_attn = use_attn
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.mode = mode
        self.pat_node_emb = None

        if word_emb == None:
            self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)
        else:
            self.embedding = torch.nn.Embedding.from_pretrained(word_emb, freeze=True)
        
        if self.use_attn:
            self.conv1 = GINEConv(nn.Linear(embedding_dim, hidden_dim), edge_dim=1)
            self.conv2 = GINEConv(nn.Linear(hidden_dim, hidden_dim), edge_dim=1)
            self.conv3 = GINEConv(nn.Linear(hidden_dim, hidden_dim), edge_dim=1)

            self.retain = nn.ModuleDict()
            for feature_key in feature_keys:
                self.retain[feature_key] = RETAINLayer(feature_size=self.max_nodes, dropout=dropout)
        else:
            self.conv1 = GINConv(Linear(embedding_dim, hidden_dim))
            self.conv2 = GINConv(Linear(hidden_dim, hidden_dim))
            self.conv3 = GINConv(Linear(hidden_dim, hidden_dim))

        self.fc = nn.Linear(hidden_dim, out_channels)


    def forward(self, node_ids, edge_index, batch, visits_cond, visits_proc, patient_id=None):
        x = self.embedding(node_ids)
        patient_indices = torch.tensor([(node_ids == patient_id[i]).nonzero(as_tuple=True)[0].item() for i in range(len(patient_id))])

        if self.use_attn:

            cond_attn = self.retain['cond'](visits_cond)
            proc_attn = self.retain['proc'](visits_proc)
            # cross_attn = self.retain['cross'](visits_cond + visits_proc)

            attn = cond_attn.add_(proc_attn)   # (batch_size, max_nodes)

            # Create a batch index tensor to map the batch index to the corresponding attention weight
            batch_index = torch.arange(attn.size(0), device=node_ids.device).repeat_interleave(torch.bincount(batch))   

            # Fill the attn_weights matrix with the correct weights using batch_index and node_ids
            attn_weights = attn[batch_index, node_ids]

            row, col = edge_index
            # Define a small constant value epsilon
            epsilon = 1e-6

            attn_weights = attn_weights / torch.max(attn_weights)
            attn_weights = torch.where(torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights)

            edge_attr = ((attn_weights[row] + epsilon) + (attn_weights[col] + epsilon)).unsqueeze(-1)


            x = F.relu(self.conv1(x, edge_index, edge_attr))
            x = F.relu(self.conv2(x, edge_index, edge_attr))
            x = F.relu(self.conv3(x, edge_index, edge_attr))

        else:
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = F.relu(self.conv3(x, edge_index))

        if self.mode == "graph":
            x = global_mean_pool(x, batch)
            x = F.dropout(x, p=0.5, training=self.training)
        elif self.mode == "node":
            patient_node_embeddings = x[patient_indices]
            x = F.dropout(patient_node_embeddings, p=0.5, training=self.training)

        logits = self.fc(x)
        return logits

    
def train(model, device, train_loader, optimizer, model_):
    model.train()
    training_loss = 0
    tot_loss = 0
    pbar= tqdm(enumerate(train_loader))
    for i, data in pbar:
        pbar.set_description(f'loss: {training_loss}')

        data = data.to(device)
        optimizer.zero_grad()
        if model_ == "GIN":
            out = model(data.x, data.edge_index, data.batch)
        elif model_ == "GINX":
            out = model(data.y, data.edge_index, data.batch)
        else:
            out = model(
                    data.y, 
                    data.edge_index, 
                    data.batch, 
                    data.visits_cond.reshape(int(train_loader.batch_size), int(len(data.visits_cond)/train_loader.batch_size), data.visits_cond.shape[1]).double(), 
                    data.visits_proc.reshape(int(train_loader.batch_size), int(len(data.visits_proc)/train_loader.batch_size), data.visits_proc.shape[1]).double(), 
                    # data.visits_drug.reshape(int(train_loader.batch_size), int(len(data.visits_drug)/train_loader.batch_size), data.visits_drug.shape[1]).double(),
                    patient_id = data.patient_id.reshape(int(train_loader.batch_size), int(len(data.patient_id)/train_loader.batch_size)).long()
                    
                )
        try:
            label = data.label.reshape(int(train_loader.batch_size), int(len(data.label)/train_loader.batch_size))
        except:
            continue
        # print(out.shape, label.shape)
        loss = F.binary_cross_entropy_with_logits(out, label.float())
        loss.backward()
        training_loss = loss
        tot_loss += loss
        optimizer.step()
    
    return tot_loss

def evaluate(model, device, loader, model_):
    model.eval()
    y_prob_all = []
    y_true_all = []

    for data in tqdm(loader):
        data = data.to(device)
        with torch.no_grad():    
            
            if model_ == "GIN":
                logits = model(data.x, data.edge_index, data.batch)
            elif model_ == "GINX":
                logits = model(data.y, data.edge_index, data.batch)
            else:
                logits = model(
                    data.y, 
                    data.edge_index, 
                    data.batch, 
                    data.visits_cond.reshape(int(loader.batch_size), int(len(data.visits_cond)/loader.batch_size), data.visits_cond.shape[1]).double(), 
                    data.visits_proc.reshape(int(loader.batch_size), int(len(data.visits_proc)/loader.batch_size), data.visits_proc.shape[1]).double(), 
                    # data.visits_drug.reshape(int(loader.batch_size), int(len(data.visits_drug)/loader.batch_size), data.visits_drug.shape[1]).double(),
                    patient_id = data.patient_id.reshape(int(loader.batch_size), int(len(data.patient_id)/loader.batch_size)).long()
                )

            y_prob = torch.sigmoid(logits)
            try:
                y_true = data.label.reshape(int(loader.batch_size), int(len(data.label)/loader.batch_size))
            except:
                continue
            y_prob_all.append(y_prob.cpu())
            y_true_all.append(y_true.cpu())
            
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)

    return y_true_all, y_prob_all

def train_loop(train_loader, val_loader, model, optimizer, device, epochs, model_, task_name):
    best_acc = 0
    best_f1 = 0
    for epoch in range(1, epochs+1):
        loss = train(model, device, train_loader, optimizer, model_)
        y_true_all, y_prob_all = evaluate(model, device, val_loader, model_)

        y_pred_all = (y_prob_all >= 0.5).astype(int)
        
        val_pr_auc = average_precision_score(y_true_all, y_prob_all, average='samples')
        val_roc_auc = roc_auc_score(y_true_all, y_prob_all, average='samples')
        val_jaccard = jaccard_score(y_true_all, y_pred_all, average='samples', zero_division=1)
        val_acc = accuracy_score(y_true_all, y_pred_all)
        val_f1 = f1_score(y_true_all, y_pred_all, average='samples', zero_division=1)
        val_precision = precision_score(y_true_all, y_pred_all, average='samples', zero_division=1)
        val_recall = recall_score(y_true_all, y_pred_all, average='samples', zero_division=1)

        if val_acc >= best_acc and val_f1 >= best_f1:
            torch.save(model.state_dict(), f'../../../data/pj20/exp_data/saved_weights_{model_}_{task_name}.pkl')
            print("best model saved")
            best_acc = val_acc
            best_f1 = val_f1

        print(f'Epoch: {epoch}, Training loss: {loss}, Val PRAUC: {val_pr_auc:.4f}, Val ROC_AUC: {val_roc_auc:.4f}, Val acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val precision: {val_precision:.4f}, Val recall: {val_recall:.4f}, Val jaccard: {val_jaccard:.4f}')


train_set = Dataset(G=G_tg, dataset=train_dataset)
val_set = Dataset(G=G_tg, dataset=val_dataset)
test_set = Dataset(G=G_tg, dataset=test_dataset)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False, drop_last=True)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False, drop_last=True)


model_ = "GraphCare"
out_channels = len(train_set[0].label)


device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

if model_ == "GIN":
    in_channels = train_set[0].x.shape[1]
    model = GIN(in_channels=in_channels, out_channels=out_channels, hidden_channels=512).to(device)
    # model = GAT(in_channels=in_channels, out_channels=1, hidden_channels=256, heads=3).to(device)
    # model = HGT(in_channels=in_channels, out_channels=out_channels, hidden_channels=512, heads=2).to(device)
elif model_ == "GINX":
    model = GINX(num_nodes=G_tg.num_nodes, embedding_dim=512, hidden_channels=512, out_channels=out_channels, word_emb=G_tg.x).to(device)

elif model_ == "GraphCare":
    # model = GINX(num_nodes=G_tg.num_nodes, embedding_dim=512, hidden_channels=512, out_channels=out_channels, word_emb=G_tg.x).to(device)
    model = GraphCare(
        num_nodes=G_tg.num_nodes - len(patient_id_set),
        feature_keys=['cond', 'proc'], 
        embedding_dim=len(G_tg.x[0]), 
        hidden_dim=512, 
        out_channels=out_channels, 
        dropout=0.5, 
        max_visits=max_visits,
        word_emb=G_tg.x,
        use_attn=False,
        mode="node",
        num_patient=len(patient_id_set),
    ).to(device)

model.double()

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
task_name = "drugrec_th015"

state_dict = torch.load(f'../../../data/pj20/exp_data/saved_weights_{model_}_{task_name}.pkl')
model.load_state_dict(state_dict)
train_loop(train_loader=train_loader, val_loader=val_loader, model=model, optimizer=optimizer, device=device, epochs=40, model_=model_, task_name=task_name)

model.load_state_dict(torch.load(f'../../../data/pj20/exp_data/saved_weights_{model_}_{task_name}.pkl'))
model.double()

y_true_all, y_prob_all = evaluate(model, device, val_loader, model_=model_)

y_pred_all = y_prob_all.copy()
y_pred_all[y_pred_all >= 0.5] = 1
y_pred_all[y_pred_all < 0.5] = 0

test_pr_auc = average_precision_score(y_true_all, y_prob_all, average="samples")
test_roc_auc = roc_auc_score(y_true_all, y_prob_all, average="samples")
test_f1 = f1_score(y_true_all, y_pred_all, average='samples')
test_jaccard = jaccard_score(y_true_all, y_pred_all, average='samples')

print(f'test PRAUC: {test_pr_auc:.4f}, test ROC_AUC: {test_roc_auc:.4f}, test F1-score: {test_f1:.4f}, test Jaccard: {test_jaccard:.4f}')