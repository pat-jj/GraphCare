import pickle
import json
import random
import networkx as nx
from pyhealth.datasets import SampleDataset
from graphcare_ import split_by_patient
from torch_geometric.utils import to_networkx, from_networkx
from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.loader import DataListLoader, DataLoader
from graphcare_ import GAT, GIN, GraphCare
from tqdm import tqdm
from pyhealth.metrics import multilabel_metrics_fn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, jaccard_score, cohen_kappa_score
import argparse
import logging
import neptune
from copy import deepcopy


def load_everything(dataset, task, kg="", kg_ratio=1.0, voc_ratio=1.0, idx=None, th="th015"):
    if kg == "GPT-KG":
        kg = ""
    if task == "drugrec" or task == "lenofstay":
        path_1 = "/data/pj20/exp_data/ccscm_ccsproc"
        path_2 = "./graphs/cond_proc/CCSCM_CCSPROC"
    elif task == "mortality" or task == "readmission":
        path_1 = "/data/pj20/exp_data/ccscm_ccsproc_atc3"
        path_2 = "./graphs/cond_proc_drug/CCSCM_CCSPROC_ATC3"

    if kg_ratio != 1.0:
        sample_dataset_file = f"{path_1}/sample_dataset_{dataset}_{task}_{kg}{th}_{kg_ratio}.pkl"
        graph_file = f"{path_1}/graph_{dataset}_{task}_{kg}{th}_{kg_ratio}.pkl"
    else:
        sample_dataset_file = f"{path_1}/sample_dataset_{dataset}_{task}_{kg}{th}.pkl"
        graph_file = f"{path_1}/graph_{dataset}_{task}_{kg}{th}.pkl"

    if voc_ratio != 1.0:
        sample_dataset_file = f"{path_1}/sample_dataset_mimic3_{task}_th015_mask_{voc_ratio}_{idx}.pkl"
        graph_file = f"{path_1}/graph_mimic3_{task}_th015_mask_{voc_ratio}_{idx}.pkl"

    map_cluster_file = f"{path_1}/clusters_{th}_ana.json" 
    map_cluster_inv = f"{path_1}/clusters_inv_{th}_ana.json"
    map_cluster_rel = f"{path_1}/clusters_rel_{th}.json"
    map_cluster_rel_inv = f"{path_1}/clusters_inv_rel_{th}.json"
    ccscm_id2clus = f"{path_1}/ccscm_id2clus_ana.json"
    ccsproc_id2clus = f"{path_1}/ccsproc_id2clus_ana.json"
    if task == "mortality" or task == "readmission":
        atc3_id2clus = f"{path_1}/atc3_id2clus_ana.json"

    ent2id_file = f"{path_2}/ent2id.json"
    rel2id_file = f"{path_2}/rel2id.json"
    ent_emb_file = f"{path_2}/entity_embedding.pkl"
    rel_emb_file = f"{path_2}/relation_embedding.pkl"


    with open(sample_dataset_file, "rb") as f:
        sample_dataset = pickle.load(f)
    with open(graph_file, "rb") as f:
        graph = pickle.load(f)
    with open(ent2id_file, "r") as f:
        ent2id = json.load(f)
    with open(rel2id_file, "r") as f:
        rel2id = json.load(f)
    with open(ent_emb_file, "rb") as f:
        ent_emb = pickle.load(f)
    with open(rel_emb_file, "rb") as f:
        rel_emb = pickle.load(f)
    with open(map_cluster_file, "r") as f:
        map_cluster = json.load(f)
    with open(map_cluster_inv, "r") as f:
        map_cluster_inv = json.load(f)
    with open(map_cluster_rel, "r") as f:
        map_cluster_rel = json.load(f)
    with open(map_cluster_rel_inv, "r") as f:
        map_cluster_rel_inv = json.load(f)
    with open(ccscm_id2clus, "r") as f:
        ccscm_id2clus = json.load(f)
    with open(ccsproc_id2clus, "r") as f:
        ccsproc_id2clus = json.load(f)
    if task == "mortality" or task == "readmission":
        with open(atc3_id2clus, "r") as f:
            atc3_id2clus = json.load(f)
    else:
        atc3_id2clus = None

    return sample_dataset, graph, ent2id, rel2id, ent_emb, rel_emb, \
                map_cluster, map_cluster_inv, map_cluster_rel, map_cluster_rel_inv, \
                    ccscm_id2clus, ccsproc_id2clus, atc3_id2clus


def get_mode_and_out_channels_and_loss_func(task, sample_dataset):
    mode = ""
    if task == "mortality" or task == "readmission":
        mode = "binary"
        out_channels = 1
        loss_function = F.binary_cross_entropy_with_logits
    elif task == "drugrec":
        mode = "multilabel"
        out_channels = len(sample_dataset[0]["drugs_ind"])
        loss_function = F.binary_cross_entropy_with_logits
    elif task == "lenofstay":
        mode = "multiclass"
        out_channels = 10
        loss_function = F.cross_entropy

    return mode, out_channels, loss_function

def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


def label_ehr_nodes(task, sample_dataset, max_nodes, ccscm_id2clus, ccsproc_id2clus, atc3_id2clus):

    for patient in tqdm(sample_dataset):
        nodes = []
        for condition in flatten(patient['conditions']):
            ehr_node = ccscm_id2clus[condition]
            nodes.append(int(ehr_node))
            if ehr_node != "2755":
                patient['node_set'].append(int(ehr_node))

        for procedure in flatten(patient['procedures']):
            ehr_node = ccsproc_id2clus[procedure]
            nodes.append(int(ehr_node))
            if ehr_node != "2755":
                patient['node_set'].append(int(ehr_node))

        if task == "mortality" or task == "readmission":
            for drug in flatten(patient['drugs']):
                ehr_node = atc3_id2clus[drug]
                nodes.append(int(ehr_node))
                if ehr_node != "2755":
                    patient['node_set'].append(int(ehr_node))

        # make one-hot encoding
        node_vec = np.zeros(max_nodes)
        node_vec[nodes] = 1
        
        patient['ehr_node_set'] = torch.tensor(node_vec)

    return sample_dataset


def get_rel_emb(map_cluster_rel):
    rel_emb = []

    for i in range(len(map_cluster_rel.keys())):
        rel_emb.append(map_cluster_rel[str(i)]['embedding'][0])

    rel_emb = np.array(rel_emb)
    return torch.tensor(rel_emb)


def get_subgraph(G, dataset, task, idx):
    patient = dataset[idx]
    while len(patient['node_set']) == 0:
        idx -= 1
        patient = dataset[idx]

    # less focused 
    # P = G.edge_subgraph(torch.tensor(patient['node_set']))

    # more focused
    # another way to get subgraph
    L = G.edge_subgraph(torch.tensor(patient['node_set']))
    P = L.subgraph(torch.tensor(patient['node_set']))

    if task == "drugrec":
        P.label = patient['drugs_ind']
    elif task == "lenofstay":
        label = np.zeros(10)
        label[patient['label']] = 1
        P.label = torch.tensor(label)
    else:
        P.label = patient['label']
    
    P.visit_padded_node = patient['visit_padded_node']
    P.ehr_nodes = patient['ehr_node_set']
    P.patient_id = patient['patient_id']
    
    return P

class Dataset(torch.utils.data.Dataset):
    def __init__(self, G, dataset, task):
        self.G = G
        self.dataset=dataset
        self.task = task
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return get_subgraph(G=self.G, dataset=self.dataset, task=self.task, idx=idx)

def get_dataloader(G_tg, train_dataset, val_dataset, test_dataset, task, batch_size):
    train_set = Dataset(G=G_tg, dataset=train_dataset, task=task)
    val_set = Dataset(G=G_tg, dataset=val_dataset, task=task)
    test_set = Dataset(G=G_tg, dataset=test_dataset, task=task)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader

    
def train(mode, patient_mode, gnn, model, device, train_loader, optimizer, loss_func):
    model.train()
    training_loss = 0
    tot_loss = 0
    pbar= tqdm(enumerate(train_loader))
    for i, data in pbar:
        pbar.set_description(f'loss: {training_loss}')

        data = data.to(device)
        optimizer.zero_grad()

        node_ids = data.y
        rel_ids = data.relation
        ehr_nodes = data.ehr_nodes.reshape(int(train_loader.batch_size), int(len(data.ehr_nodes)/train_loader.batch_size)).float() if patient_mode != "graph" else None
        visit_node = data.visit_padded_node.reshape(int(train_loader.batch_size), int(len(data.visit_padded_node)/train_loader.batch_size), data.visit_padded_node.shape[1]).float() 
        out = model(
                node_ids = node_ids, 
                rel_ids = rel_ids,
                edge_index = data.edge_index,
                batch = data.batch,
                visit_node = visit_node, 
                ehr_nodes = ehr_nodes,
                in_drop=True,
            )

        label = data.label.reshape(int(train_loader.batch_size), int(len(data.label)/train_loader.batch_size))

        loss = loss_func(out, label.float())
        loss.backward()
        training_loss = loss
        tot_loss += loss
        optimizer.step()
    
    return tot_loss

def evaluate(mode, patient_mode, gnn, model, device, loader):
    model.eval()
    y_prob_all = []
    y_true_all = []

    for data in tqdm(loader):
        data = data.to(device)
        with torch.no_grad():    

            node_ids = data.y
            rel_ids = data.relation
            ehr_nodes = data.ehr_nodes.reshape(int(loader.batch_size), int(len(data.ehr_nodes)/loader.batch_size)).float() if patient_mode != "graph" else None
            visit_node = data.visit_padded_node.reshape(int(loader.batch_size), int(len(data.visit_padded_node)/loader.batch_size), data.visit_padded_node.shape[1]).float() 
            model
            logits = model(
                    node_ids = node_ids, 
                    rel_ids = rel_ids,
                    edge_index = data.edge_index,
                    batch = data.batch,
                    visit_node = visit_node,
                    ehr_nodes = ehr_nodes,
                )

            if mode == "multiclass":
                y_prob = F.softmax(logits, dim=-1)
            else:
                y_prob = torch.sigmoid(logits)
            
            y_true = data.label.reshape(int(loader.batch_size), int(len(data.label)/loader.batch_size))

            y_prob_all.append(y_prob.cpu())
            y_true_all.append(y_true.cpu())
            
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)

    return y_true_all, y_prob_all

def train_loop(dataset, task, mode, patient_mode, gnn, train_loader, val_loader, model, optimizer, loss_func, device, epochs, logger=None, run=None, early_stop=3):
    best_roc_auc = 0
    best_f1 = 0
    early_stop_indicator = 0
    for epoch in range(1, epochs+1):
        loss = train(mode, patient_mode, gnn, model, device, train_loader, optimizer, loss_func)
        y_true_all, y_prob_all = evaluate(mode, patient_mode, gnn, model, device, val_loader)
        
        if mode == "binary":
            y_pred_all = (y_prob_all >= 0.5).astype(int)

            val_pr_auc = average_precision_score(y_true_all, y_prob_all)
            val_roc_auc = roc_auc_score(y_true_all, y_prob_all)
            val_jaccard = jaccard_score(y_true_all, y_pred_all, average="macro", zero_division=1)
            val_acc = accuracy_score(y_true_all, y_pred_all)
            val_f1 = f1_score(y_true_all, y_pred_all, average="macro", zero_division=1)
            val_precision = precision_score(y_true_all, y_pred_all, average="macro", zero_division=1)
            val_recall = recall_score(y_true_all, y_pred_all, average="macro", zero_division=1)
        elif mode == "multilabel":
            y_pred_all = (y_prob_all >= 0.5).astype(int)

            val_pr_auc = average_precision_score(y_true_all, y_prob_all, average="samples")
            val_roc_auc = roc_auc_score(y_true_all, y_prob_all, average="samples")
            val_jaccard = jaccard_score(y_true_all, y_pred_all, average="samples", zero_division=1)
            val_acc = accuracy_score(y_true_all, y_pred_all)
            val_f1 = f1_score(y_true_all, y_pred_all, average="samples", zero_division=1)
            val_precision = precision_score(y_true_all, y_pred_all, average="samples", zero_division=1)
            val_recall = recall_score(y_true_all, y_pred_all, average="samples", zero_division=1)
        elif mode == "multiclass":
            y_pred_all = np.argmax(y_prob_all, axis=-1)
            y_true_all = np.argmax(y_true_all, axis=-1)

            val_pr_auc = 0
            val_roc_auc = roc_auc_score(y_true_all, y_prob_all, multi_class="ovr", average="weighted")
            val_jaccard = cohen_kappa_score(y_true_all, y_pred_all)
            val_acc = accuracy_score(y_true_all, y_pred_all)
            val_f1 = f1_score(y_true_all, y_pred_all, average="weighted")
            val_precision = 0
            val_recall = 0

        if val_roc_auc >= best_roc_auc:
            torch.save(model.state_dict(), f'../../../data/pj20/exp_data/saved_weights_{dataset}_{task}_{model.gnn}.pkl')
            print("best model saved")
            best_roc_auc = val_roc_auc
            early_stop_indicator = 0
            # best_f1 = val_f1
        else:
            early_stop_indicator += 1
            if early_stop_indicator >= early_stop:
                break
        if run is not None:
            run["train/loss"].append(loss)
            run["val/pr_auc"].append(val_pr_auc)
            run["val/roc_auc"].append(val_roc_auc)
            run["val/acc"].append(val_acc)
            run["val/f1"].append(val_f1)
            run["val/precision"].append(val_precision)
            run["val/recall"].append(val_recall)
            run["val/jaccard"].append(val_jaccard)

        print(f'Epoch: {epoch}, Training loss: {loss}, Val PRAUC: {val_pr_auc:.4f}, Val ROC_AUC: {val_roc_auc:.4f}, Val acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val precision: {val_precision:.4f}, Val recall: {val_recall:.4f}, Val jaccard: {val_jaccard:.4f}')
        if logger is not None:
            logger.info(f'Epoch: {epoch}, Training loss: {loss}, Val PRAUC: {val_pr_auc:.4f}, Val ROC_AUC: {val_roc_auc:.4f}, Val acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val precision: {val_precision:.4f}, Val recall: {val_recall:.4f}, Val jaccard: {val_jaccard:.4f}')


def construct_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mimic3')
    parser.add_argument('--task', type=str, default='mortality')
    parser.add_argument('--kg', type=str, default='GPT-KG')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--decay_rate', type=float, default=0.01)
    parser.add_argument('--freeze_emb', type=str, default="False")
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--patient_mode', type=str, default='joint', choices=['joint', 'graph', 'node'])
    parser.add_argument('--alpha', type=str, default="True", choices=["True", "False"])
    parser.add_argument('--beta', type=str, default="True", choices=["True", "False"])
    parser.add_argument('--edge_attn', type=str, default="True", choices=["True", "False"])
    parser.add_argument('--self_attn', type=float, default=0.)
    parser.add_argument("--gnn", type=str, default="BAT", choices=["GAT", "BAT", "GIN"])
    parser.add_argument('--hyperparameter_search', type=bool, default=False)
    parser.add_argument('--attn_init', type=str, default="False", choices=["True", "False"])
    parser.add_argument('--in_drop_rate', type=float, default=0.)
    parser.add_argument('--out_drop_rate', type=float, default=0.)
    parser.add_argument('--kg_ratio', type=float, default=1.0)
    parser.add_argument('--ehr_feat_ratio', type=float, default=1.0)

    args = parser.parse_args()
    return args


def get_logger(dataset, task, kg, hidden_dim, epochs, lr, decay_rate, dropout, num_layers):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    file_handler = logging.FileHandler(f'./training_logs/{dataset}_{task}_{kg}_{hidden_dim}_{epochs}_{lr}_{decay_rate}_{dropout}_{num_layers}.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

def single_run(args, params):

    dataset, task, kg, batch_size, hidden_dim, epochs, lr, weight_decay, dropout, num_layers, decay_rate, gnn, patient_mode, alpha, beta, edge_attn, freeze, attn_init, in_drop_rate, kg_ratio, train_ratio, voc_ratio, data_idx = \
        params['dataset'], params['task'], params['kg'], params['batch_size'], params['hidden_dim'], params['epochs'], params['lr'], params['weight_decay'], params['dropout'], params['num_layers'], params['decay_rate'], params['gnn'], params['patient_mode'], params['alpha'], params['beta'], params['edge_attn'], params['freeze'], params['attn_init'], params['in_drop_rate'], params['kg_ratio'], params['train_ratio'], params['voc_ratio'], params['data_idx']
     
    run = neptune.init_run(
        project="patrick.jiang.cs/GraphCare",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNDFjZWU1ZC1mZGM5LTQ2MTItYTk3ZC02ODIzOTA4MTY0YmIifQ==",
    )

    run["parameters"] = params
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else 'cpu')
    logger = get_logger(dataset, task, kg, hidden_dim, epochs, lr, decay_rate, dropout, num_layers)

    print("device:", device)

    # load dataset
    sample_dataset, G, ent2id, rel2id, ent_emb, rel_emb, \
                map_cluster, map_cluster_inv, map_cluster_rel, map_cluster_rel_inv, \
                    ccscm_id2clus, ccsproc_id2clus, atc3_id2clus = load_everything(dataset, task, kg, kg_ratio, voc_ratio=voc_ratio,idx=data_idx)
    mode, out_channels, loss_function = get_mode_and_out_channels_and_loss_func(task=task, sample_dataset=sample_dataset)


    for i in range(len(sample_dataset)):
        sample_dataset[i]['visit_padded_node'] = torch.tensor(np.append(sample_dataset[i]['visit_padded_node'].numpy(), 0)).reshape(1, -1)
    # label direct ehr node
    print("Labeling direct ehr nodes...")
    sample_dataset = label_ehr_nodes(task, sample_dataset, len(map_cluster), ccscm_id2clus, ccsproc_id2clus, atc3_id2clus)
    print("Splitting dataset...")
    train_dataset, val_dataset, test_dataset = split_by_patient(sample_dataset, [0.8, 0.1, 0.1], train_ratio=train_ratio, seed=528)
    # if feat_ratio != 1.0:
        # with open(f'/data/pj20/exp_data/ccscm_ccsproc/val_dataset_mimic3_{task}_th015_{feat_ratio}.pkl', 'rb') as f:
        #     val_dataset = pickle.load(f)
        #     val_dataset = label_ehr_nodes(task, val_dataset, len(map_cluster), ccscm_id2clus, ccsproc_id2clus, atc3_id2clus)
        # with open(f'/data/pj20/exp_data/ccscm_ccsproc/train_dataset_mimic3_{task}_th015_{feat_ratio}.pkl', 'rb') as f:
        #     train_dataset = pickle.load(f)
        #     train_dataset = label_ehr_nodes(task, train_dataset, len(map_cluster), ccscm_id2clus, ccsproc_id2clus, atc3_id2clus)
    # get initial node attention
    print("Getting initial node attention...")
    if task == "mortality" or task == "readmission":
        attn_file = f"/data/pj20/exp_data/ccscm_ccsproc_atc3/attention_weights_{task}.pkl"
    elif task == "lenofstay" or task == "drugrec":
        attn_file = f"/data/pj20/exp_data/ccscm_ccsproc/attention_weights_{task}.pkl"
    else:
        raise NotImplementedError
    
    with open(attn_file, "rb") as f:
        attn_weights = np.append(pickle.load(f), [0]).reshape(-1,1)
        attn_weights = torch.tensor(attn_weights)

    G.add_nodes_from([
    (2755, {'y': int(2755), 'x': np.zeros((1, ent_emb.shape[1])).tolist()[0]})
    ])
    print(G)

    G_tg = from_networkx(G)

    # get embedding
    print("Getting embedding...")
    rel_emb = get_rel_emb(map_cluster_rel)
    node_emb = G_tg.x 

    num_nodes=node_emb.shape[0]
    num_rels=rel_emb.shape[0]
    max_visit=sample_dataset[0]['visit_padded_node'].shape[0]

    # get dataloader
    print("Getting dataloader...")
    train_loader, val_loader, test_loader = get_dataloader(G_tg, train_dataset, val_dataset, test_dataset, task, batch_size)
    
    # get model
    print("Getting model...")
    model = GraphCare(
        num_nodes=num_nodes,
        num_rels=num_rels,
        max_visit=max_visit,
        embedding_dim=node_emb.shape[1],
        hidden_dim=hidden_dim,
        out_channels=out_channels,
        layers=num_layers,
        dropout=dropout,
        decay_rate=decay_rate,
        node_emb=node_emb,
        rel_emb=rel_emb,
        patient_mode=patient_mode,
        use_alpha=True if alpha == "True" else False,
        use_beta=True if beta == "True" else False,
        use_edge_attn=True if edge_attn == "True" else False,
        gnn=gnn,
        freeze=True if freeze == "True" else False,
        attn_init=attn_weights if attn_init == "True" else None,
        drop_rate=in_drop_rate,
    )
    model.to(device)

    print(model)

    # train
    logger.info("Start training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loop(
        dataset=dataset,
        task=task,
        mode=mode,
        patient_mode=patient_mode,
        gnn=gnn, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        model=model, 
        optimizer=optimizer, 
        loss_func=loss_function, 
        device=device, 
        epochs=epochs, 
        logger=logger, 
        run=run
        )

    run.stop()


def hyper_search_(args, params):
    hyperparameter_options = {
        # 'batch_size': [16, 32, 64],
        # 'hidden_dim': [128, 256, 512],
        # 'lr': [0.001, 0.0001, 0.00001],
        # 'weight_decay': [0.001, 0.0001, 0.00001],
        # 'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
        # 'num_layers': [1, 2, 3, 4],
        # 'decay_rate': [0.01, 0.02, 0.03],
        # 'patient_mode': [
        #     "joint", 
        #     "graph", 
        #     "node"
        #     ],
        # 'gnn' : [
        #     "GAT", 
        #     "GIN", 
        #     "BAT"
        #     ],
        # 'edge_attn': [True, False]
        # "in_drop_rate":[
        #     0.1, 
        #     0.2, 
        #     0.3, 
        #     0.5, 
        #     0.7, 
        #     0.9
        # ]
        # "kg_ratio":[]
            # 0.1,
            # 0.3,
            # 0.5,
            # 0.7,
            # 0.9
        # ]
        # "train_ratio": [
        #     0.001,
        #     0.002,
        #     0.003,
        #     0.004,
        #     0.005,
        #     0.006,
        #     0.007,
        #     0.008,
        #     0.009,
        #     0.01,
        #     0.02,
        #     0.03,
        #     0.04,
        #     0.05,
        #     0.06,
        #     0.07,
        #     0.08,
        #     0.09,
        #     0.1,
        #     0.3,
        #     0.5,
        #     0.7,
        #     0.9,
        # ],
        "voc_ratio": [
            0.1,
            0.2,
            0.3,
            # 0.4,
            # 0.5,
            # 0.6,
            # 0.7,
            # 0.8,
            # 0.9,
        ],

    }

    data_idxs = [
        0, 
        1, 
        2
        ]

    for data_idx in data_idxs:
        for hp_name, hp_options in hyperparameter_options.items():
            print(f"now searching for {hp_name}...")
            for hp_value in hp_options:
                print(f"now searching for {hp_name}={hp_value}...")
                params_copy = params.copy()
                params_copy[hp_name] = hp_value
                params_copy['data_idx'] = data_idx
                for i in range(1):
                    single_run(args, params_copy)


def main():
    args = construct_args()
    dataset, task, kg, batch_size, hidden_dim, epochs, lr, weight_decay, dropout, num_layers, \
     decay_rate, patient_mode, alpha, beta, edge_attn, gnn, hyper_search, freeze, attn_init, in_drop_rate, kg_ratio = \
        args.dataset, args.task, args.kg, args.batch_size, args.hidden_dim, args.epochs, args.lr, args.weight_decay, \
            args.dropout, args.num_layers, args.decay_rate, args.patient_mode, args.alpha, args.beta, args.edge_attn, args.gnn, args.hyperparameter_search, args.freeze_emb, args.attn_init, args.in_drop_rate, args.kg_ratio

    parameters = {
        "dataset": dataset,
        "task": task,
        "kg": kg,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "num_layers": num_layers,
        "decay_rate": decay_rate,
        "patient_mode": patient_mode,
        "alpha": alpha,
        "beta": beta,
        "edge_attn": edge_attn,
        "gnn": gnn,
        "freeze": freeze,
        "attn_init": attn_init,
        "in_drop_rate": in_drop_rate,
        "kg_ratio": kg_ratio,
        "train_ratio": 1.0,
        "feat_ratio": 1.0
    }
    
    print(parameters)

    if hyper_search:
        # hyperparameter search
        print("Hyperparameter search...")
        hyper_search_(args, parameters)

    else:
        single_run(args, parameters)


if __name__ == '__main__':
    main()