import csv
from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset
from GraphCare.task_fn import drug_recommendation_fn, drug_recommendation_mimic4_fn, mortality_prediction_mimic3_fn, readmission_prediction_mimic3_fn, length_of_stay_prediction_mimic3_fn, length_of_stay_prediction_mimic4_fn, mortality_prediction_mimic4_fn, readmission_prediction_mimic4_fn
import pickle
import json
from pyhealth.tokenizer import Tokenizer
import numpy as np
from tqdm import tqdm
import torch
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import networkx as nx
from torch_geometric.utils import to_networkx, from_networkx


def load_dataset(load_processed_dataset, dataset, task):
    if task == "drugrec":
        file_name = f'/data/pj20/exp_data/ccscm_ccsproc/sample_dataset_{dataset}_{task}_th015.pkl'
    elif task == "mortality" or task == "readmission" or task == "lenofstay":        
        file_name = f'/data/pj20/exp_data/ccscm_ccsproc_atc3/sample_dataset_{dataset}_{task}_th015.pkl'

    if load_processed_dataset:
        ### load processed dataset
        print("load processed dataset ...")

        with open(file_name, 'rb') as f:
                sample_dataset = pickle.load(f)
            
    else:
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
                if row['level'] == '3.0':
                    drug_dict[row['code']] = row['name'].lower()



        if dataset == "mimic3":
            ds = MIMIC3Dataset(
            root="/data/physionet.org/files/mimiciii/1.4/", 
            tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],      
            code_mapping={
                "NDC": ("ATC", {"target_kwargs": {"level": 3}}),
                "ICD9CM": "CCSCM",
                "ICD9PROC": "CCSPROC"
                },
            )
        elif dataset == "mimic4":
            ds = MIMIC4Dataset(
            root="/data/physionet.org/files/mimiciv/2.0/hosp/", 
            tables=["diagnoses_icd", "procedures_icd", "prescriptions"],      
            code_mapping={
                "NDC": ("ATC", {"target_kwargs": {"level": 3}}),
                "ICD9CM": "CCSCM",
                "ICD9PROC": "CCSPROC",
                "ICD10CM": "CCSCM",
                "ICD10PROC": "CCSPROC",
                },
            dev=False
            )

        if task == "drugrec":
            if dataset == "mimic3":
                sample_dataset = ds.set_task(drug_recommendation_fn)
            if dataset == "mimic4":
                sample_dataset = ds.set_task(drug_recommendation_mimic4_fn)
        elif task == "mortality":
            if dataset == "mimic3":
                sample_dataset = ds.set_task(mortality_prediction_mimic3_fn)
            if dataset == "mimic4":
                sample_dataset = ds.set_task(mortality_prediction_mimic4_fn)
        elif task == "readmission":
            if dataset == "mimic3":
                sample_dataset = ds.set_task(readmission_prediction_mimic3_fn)
            if dataset == "mimic4":
                sample_dataset = ds.set_task(readmission_prediction_mimic4_fn)
        elif task == "lenofstay":
            if dataset == "mimic3":
                sample_dataset = ds.set_task(length_of_stay_prediction_mimic3_fn)
            elif dataset == "mimic4":
                sample_dataset = ds.set_task(length_of_stay_prediction_mimic4_fn)
    
    return sample_dataset


def load_mappings():
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
            if row['level'] == '3.0':
                drug_dict[row['code']] = row['name'].lower()

    return condition_dict, procedure_dict, drug_dict


def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


def load_embeddings(task):
    if task == "drugrec" or task == "lenofstay":
        with open('./graphs/cond_proc/CCSCM_CCSPROC/ent2id.json', 'r') as file:
            ent2id = json.load(file)
        with open('./graphs/cond_proc/CCSCM_CCSPROC/rel2id.json', 'r') as file:
            rel2id = json.load(file)
        with open('./graphs/cond_proc/CCSCM_CCSPROC/entity_embedding.pkl', 'rb') as file:
            ent_emb = pickle.load(file)
        with open('./graphs/cond_proc/CCSCM_CCSPROC/relation_embedding.pkl', 'rb') as file:
            rel_emb = pickle.load(file)

    elif task == "mortality" or task == "readmission":
        with open('./graphs/cond_proc_drug/CCSCM_CCSPROC_ATC3/ent2id.json', 'r') as file:
            ent2id = json.load(file)
        with open('./graphs/cond_proc_drug/CCSCM_CCSPROC_ATC3/rel2id.json', 'r') as file:
            rel2id = json.load(file)
        with open('./graphs/cond_proc_drug/CCSCM_CCSPROC_ATC3/entity_embedding.pkl', 'rb') as file:
            ent_emb = pickle.load(file)
        with open('./graphs/cond_proc_drug/CCSCM_CCSPROC_ATC3/relation_embedding.pkl', 'rb') as file:
            rel_emb = pickle.load(file)

    return ent2id, rel2id, ent_emb, rel_emb


def multihot(label, num_labels):
    multihot = np.zeros(num_labels)
    for l in label:
        multihot[l] = 1
    return multihot


def prepare_label(sample_dataset, drugs):
    label_tokenizer = Tokenizer(
        sample_dataset.get_all_tokens(key='drugs')
    )

    labels_index = label_tokenizer.convert_tokens_to_indices(drugs)
    num_labels = label_tokenizer.get_vocabulary_size()
    labels = multihot(labels_index, num_labels)
    return labels


def prepare_drug_indices(sample_dataset):
    for patient in tqdm(sample_dataset):
        patient['drugs_ind'] = torch.tensor(prepare_label(sample_dataset, patient['drugs']))
    return sample_dataset


def clustering(task, ent_emb, rel_emb, threshold=0.15, load_cluster=False, save_cluster=False):
    if task == "drugrec" or task == "lenofstay":
        path = "/data/pj20/exp_data/ccscm_ccsproc"
    else:
        path = "/data/pj20/exp_data/ccscm_ccsproc_atc3"

    if load_cluster:
        with open(f'{path}/clusters_th015.json', 'r', encoding='utf-8') as f:
            map_cluster = json.load(f)
        with open(f'{path}/clusters_inv_th015.json', 'r', encoding='utf-8') as f:
            map_cluster_inv = json.load(f)
        with open(f'{path}/clusters_rel_th015.json', 'r', encoding='utf-8') as f:
            map_cluster_rel = json.load(f)
        with open(f'{path}/clusters_inv_rel_th015.json', 'r', encoding='utf-8') as f:
            map_cluster_inv_rel = json.load(f)

    else:
        cluster_alg = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, linkage='average', affinity='cosine')
        cluster_labels = cluster_alg.fit_predict(ent_emb)
        cluster_labels_rel = cluster_alg.fit_predict(rel_emb)

        def nested_dict():
            return defaultdict(list)

        map_cluster = defaultdict(nested_dict)

        for unique_l in np.unique(cluster_labels):
            for cur in range(len(cluster_labels)):
                if cluster_labels[cur] == unique_l:
                    map_cluster[str(unique_l)]['nodes'].append(cur)

        for unique_l in map_cluster.keys():
            nodes = map_cluster[unique_l]['nodes']
            nodes = np.array(nodes)
            embedding_mean = np.mean(ent_emb[nodes], axis=0)
            map_cluster[unique_l]['embedding'].append(embedding_mean.tolist())

        map_cluster_inv = {}
        for cluster_label, item in map_cluster.items():
            for node in item['nodes']:
                map_cluster_inv[str(node)] = cluster_label

        map_cluster_rel = defaultdict(nested_dict)

        for unique_l in np.unique(cluster_labels_rel):
            for cur in range(len(cluster_labels_rel)):
                if cluster_labels_rel[cur] == unique_l:
                    map_cluster_rel[str(unique_l)]['relations'].append(cur)

        for unique_l in map_cluster_rel.keys():
            nodes = map_cluster_rel[unique_l]['relations']
            nodes = np.array(nodes)
            embedding_mean = np.mean(ent_emb[nodes], axis=0)
            map_cluster_rel[unique_l]['embedding'].append(embedding_mean.tolist())

        map_cluster_inv_rel = {}
        for cluster_label, item in map_cluster_rel.items():
            for node in item['relations']:
                map_cluster_inv_rel[str(node)] = cluster_label

        if save_cluster:
            with open(f'{path}/clusters_th015.json', 'w', encoding='utf-8') as f:
                json.dump(map_cluster, f, indent=6)
            with open(f'{path}/clusters_inv_th015.json', 'w', encoding='utf-8') as f:
                json.dump(map_cluster_inv, f, indent=6)
            with open(f'{path}/clusters_rel_th015.json', 'w', encoding='utf-8') as f:
                json.dump(map_cluster_rel, f, indent=6)
            with open(f'{path}/clusters_inv_rel_th015.json', 'w', encoding='utf-8') as f:
                json.dump(map_cluster_inv_rel, f, indent=6)
        
    return map_cluster, map_cluster_inv, map_cluster_rel, map_cluster_inv_rel


def process_graph(dataset, task, sample_dataset, ent2id, rel2id, map_cluster, map_cluster_inv, map_cluster_rel, map_cluster_inv_rel, save_graph=False):
    if task == "drugrec" or task == "lenofstay":
        path = "/data/pj20/exp_data/ccscm_ccsproc"
    else:
        path = "/data/pj20/exp_data/ccscm_ccsproc_atc3"

    G = nx.Graph()

    for cluster_label, item in map_cluster.items():
        G.add_nodes_from([
            (int(cluster_label), {'y': int(cluster_label), 'x': item['embedding'][0]})
        ])

    for patient in tqdm(sample_dataset):
        triple_set = set()
        conditions = flatten(patient['conditions'])
        for condition in conditions:
            cond_file = f'./graphs/condition/CCSCM/{condition}.txt'
            with open(cond_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                # in case the map and emb is not up-to-date
                try:
                    items = line.split('\t')
                    if len(items) == 3:
                        h, r, t = items
                        t = t[:-1]
                        h = ent2id[h]
                        r = rel2id[r]
                        t = ent2id[t]
                        triple = (h, r, t)
                        if triple not in triple_set:
                            edge = (int(map_cluster_inv[h]), int(map_cluster_inv[t]))
                            G.add_edge(*edge, relation=int(map_cluster_inv_rel[r]))
                        triple_set.add(triple)
                except:
                    continue
        
        procedures = flatten(patient['procedures'])
        for procedure in procedures:
            proc_file = f'./graphs/procedure/CCSPROC/{procedure}.txt'
            with open(proc_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                try:
                    items = line.split('\t')
                    if len(items) == 3:
                        h, r, t = items
                        t = t[:-1]
                        h = ent2id[h]
                        r = rel2id[r]
                        t = ent2id[t]
                        triple = (h, r, t)
                        if triple not in triple_set:
                            edge = (int(map_cluster_inv[h]), int(map_cluster_inv[t]))
                            G.add_edge(*edge, relation=int(map_cluster_inv_rel[r]))
                            triple_set.add(triple)   
                except:
                    continue

        if task == "mortality" or task == "readmission":
            drugs = flatten(patient['drugs'])
            for drug in drugs:
                drug_file = f'./graphs/drug/ATC3/{drug}.txt'

                with open(drug_file, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    try:
                        items = line.split('\t')
                        if len(items) == 3:
                            h, r, t = items
                            t = t[:-1]
                            h = ent2id[h]
                            r = rel2id[r]
                            t = ent2id[t]
                            triple = (h, r, t)
                            if triple not in triple_set:
                                edge = (int(map_cluster_inv[h]), int(map_cluster_inv[t]))
                                G.add_edge(*edge, relation=int(map_cluster_inv_rel[r]))
                                triple_set.add(triple)
                    except:
                        continue

    if save_graph:
        with open(f'{path}/graph_{dataset}_{task}_th015.pkl', 'wb') as f:
            pickle.dump(G, f)

    return G

def pad_and_convert(visits, max_visits, max_nodes):
    padded_visits = []
    for idx in range(len(visits)-1, -1, -1):
        visit_multi_hot = torch.zeros(max_nodes)
        for idx, med_code in enumerate(visits[idx]):
            visit_multi_hot[med_code] = 1
        padded_visits.append(visit_multi_hot)
    while len(padded_visits) < max_visits:
        padded_visits.append(torch.zeros(max_nodes))
    return torch.stack(padded_visits, dim=0)


def process_sample_dataset(dataset, task, sample_dataset, G_tg, ent2id, rel2id, map_cluster, map_cluster_inv, map_cluster_rel, map_cluster_inv_rel, save_dataset=False):
    if task == "drugrec" or task == "lenofstay":
        path = "/data/pj20/exp_data/ccscm_ccsproc"
    else:
        path = "/data/pj20/exp_data/ccscm_ccsproc_atc3"

    c_v = []
    for patient in sample_dataset:
        c_v.append(len(patient['conditions']))

    max_visits = max(c_v)      

    for patient in tqdm(sample_dataset):
        node_set_all = set()
        node_set_list = []
        for visit_i in range(len(patient['conditions'])):
            triple_set = set()
            node_set = set() 
            conditions = patient['conditions'][visit_i]
            procedures = patient['procedures'][visit_i]
            if task == "mortality" or task == "readmission":
                drugs = patient['drugs'][visit_i]

            for condition in conditions:
                cond_file = f'./graphs/condition/CCSCM/{condition}.txt'
                with open(cond_file, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    try:
                        items = line.split('\t')
                        if len(items) == 3:
                            h, r, t = items
                            t = t[:-1]
                            h = ent2id[h]
                            # r = int(rel2id[r]) + len(ent_emb)
                            t = ent2id[t]
                            triple = (h, r, t)
                            if triple not in triple_set:
                                triple_set.add(triple)
                                node_set.add(int(map_cluster_inv[h]))
                                # node_set.add(r)
                                node_set.add(int(map_cluster_inv[t]))
                    except:
                        continue

            for procedure in procedures:
                proc_file = f'./graphs/procedure/CCSPROC/{procedure}.txt'
                with open(proc_file, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    try:
                        items = line.split('\t')
                        if len(items) == 3:
                            h, r, t = items
                            t = t[:-1]
                            h = ent2id[h]
                            # r = int(rel2id[r]) + len(ent_emb)
                            t = ent2id[t]
                            triple = (h, r, t)
                            if triple not in triple_set:
                                triple_set.add(triple)
                                node_set.add(int(map_cluster_inv[h]))
                                # node_set.add(r)
                                node_set.add(int(map_cluster_inv[t]))
                    except:
                        continue

            if task == "mortality" or task == "readmission":
                for drug in drugs:
                    drug_file = f'./graphs/drug/ATC3/{drug}.txt'

                    with open(drug_file, 'r') as f:
                        lines = f.readlines()

                    for line in lines:
                        try:
                            items = line.split('\t')
                            if len(items) == 3:
                                h, r, t = items
                                t = t[:-1]
                                h = ent2id[h]
                                # r = int(rel2id[r]) + len(ent_emb)
                                t = ent2id[t]
                                triple = (h, r, t)
                                if triple not in triple_set:
                                    triple_set.add(triple)
                                    node_set.add(int(map_cluster_inv[h]))
                                    # node_set.add(r)
                                    node_set.add(int(map_cluster_inv[t]))
                        except:
                            continue

            node_set_list.append([*node_set])
            node_set_all.update(node_set)

        padded_visits = pad_and_convert(node_set_list, max_visits, len(G_tg.x))
        patient['node_set'] = [*node_set_all]
        patient['visit_padded_node'] = padded_visits


    if save_dataset:
        with open(f'{path}/sample_dataset_{dataset}_{task}_th015.pkl', 'wb') as f:
            pickle.dump(sample_dataset, f)

    return sample_dataset


def run(dataset, task):
    if task == "drugrec":
        load_processed_dataset = False
    else:
        load_processed_dataset = False
    load_cluster = True
    save_cluster = False
    save_graph = True
    save_processed_dataset = True

    print(f"Dataset: {dataset}, Task: {task}")
    print(f"Load processed dataset: {load_processed_dataset}")
    print(f"Load cluster: {load_cluster}")
    print(f"Save cluster: {save_cluster}")
    print(f"Save graph: {save_graph}")
    print(f"Save processed dataset: {save_processed_dataset}")

    print("Loading dataset...")
    sample_dataset = load_dataset(load_processed_dataset, dataset=dataset, task=task)

    print("Loading embeddings...")
    ent2id, rel2id, ent_emb, rel_emb = load_embeddings(task)

    if task == "drugrec" and not load_processed_dataset:
        print("Preparing drug indices...")
        sample_dataset = prepare_drug_indices(sample_dataset)

    print("Clustering...")
    map_cluster, map_cluster_inv, map_cluster_rel, map_cluster_inv_rel = clustering(task, ent_emb, rel_emb, threshold=0.15, load_cluster=load_cluster, save_cluster=save_cluster)

    print("Processing graph...")
    G = process_graph(dataset, task, sample_dataset, ent2id, rel2id, map_cluster, map_cluster_inv, map_cluster_rel, map_cluster_inv_rel, save_graph=save_graph)
    G_tg = from_networkx(G)

    print("Processing dataset...")
    sample_dataset = process_sample_dataset(dataset, task, sample_dataset, G_tg, ent2id, rel2id, map_cluster, map_cluster_inv, map_cluster_rel, map_cluster_inv_rel, save_dataset=save_processed_dataset)


def main():
    datasets = [
        # "mimic3", 
        "mimic4"
        ]
    tasks = [
        # "drugrec", 
        "mortality", 
        "readmission", 
        "lenofstay"
        ]

    for dataset in datasets:
        for task in tasks:
            run(dataset, task)


if __name__ == "__main__":
    main()
