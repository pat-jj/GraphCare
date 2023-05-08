import json
from collections import defaultdict
import glob
from tqdm import tqdm
import random

import json
import numpy as np

with open('../graphs/condition/ICD9CM_base/ent2id.json', 'r') as f:
    ent2id = json.load(f)

with open('../graphs/condition/ICD9CM_base/rel2id.json', 'r') as f:
    rel2id = json.load(f)

ent_emb = np.load('../KG_mapping/umls/ent_emb.npy')

with open('../KG_mapping/ICD9CM_to_UMLS.csv', 'r') as f:
    lines = f.readlines()
    lines = lines[1:]

with open('../KG_mapping/umls/umls.csv', 'r') as f:
    lines_1 = f.readlines()

triple_set = set()
tuple_set = set()

for line in lines_1:
    items = line.split('\t')
    e1 = items[1]
    r = items[0]
    e2 = items[2]
    triple_set.add((e1, r, e2))

icd9_to_umls = {}
for line in lines:
    icd9cm, umls = line.split(',')
    umls = umls[:-1]
    icd9_to_umls[icd9cm.replace('.', '')] = umls

with open('../graphs/condition/ICD9CM_base/id2ent_new.json', 'r') as f:
    id2ent_new = json.load(f)

triple_files = glob.glob('../graphs/condition/ICD9CM_base/*.txt')
store_dir = "/data/pj20/graphs/umls_icd9_2hop/"


## main process
global_node_triple_store = defaultdict(list)
searched_node = set()
global_node_set = set()
for i in tqdm(range(int(len(triple_files) ))):
    triple_file = triple_files[i]
    with open(triple_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        h, r, t = line[:-1].split('\t')
        global_node_set.add(h)
        global_node_set.add(t)


# Create a dictionary that maps each node to the triples that contain it
node_triples = {node: [] for node in global_node_set}
for triple in triple_set:
    for node in triple:
        if node in node_triples:
            node_triples[node].append(triple)

# Randomly select triples for each node
for node in tqdm(global_node_set):
    sample_size = min(len(node_triples[node]), 100000) # set the sample size to be the minimum of the desired number of triples and the actual number of triples available
    random_triple_set = random.sample(node_triples[node], sample_size)
    for triple in random_triple_set:
        if len(global_node_triple_store[node]) >= 3:
            break
        if node == triple[0]:
            global_node_triple_store[node].append(triple)
            if triple[2] in global_node_set:
                global_node_triple_store[triple[2]].append(triple)
        if node == triple[2]:
            global_node_triple_store[node].append(triple)
            if triple[0] in global_node_set:
                global_node_triple_store[triple[0]].append(triple)

with open('./global_node_triple_store.json', 'w') as f:
    json.dump(global_node_triple_store, f)
    

for i in tqdm(range(int(len(triple_files) ))):
    triple_file = triple_files[i]
    triple_set_ = set()
    node_triple_dict = defaultdict(list)
    out_file = store_dir + triple_file.replace('../graphs/condition/ICD9CM_base/', '')

    with open(triple_file, 'r') as f:
        lines = f.readlines()
        out_str = f.read()

    for line in lines:
        h, r, t = line[:-1].split('\t')
        triple = (h, r, t)
        triple_set_.add(triple)

    for triple in triple_set_: 
        h, r, t = triple
        if h in global_node_set:
            node_triple_dict[h].append(triple)
        if t in global_node_set:
            node_triple_dict[t].append(triple)
    
    for node in node_triple_dict:
        for triple in global_node_triple_store[node]:
            if triple not in node_triple_dict[node]:
                node_triple_dict[node].append(triple)
    
    for node in node_triple_dict:
        for triple in node_triple_dict[node]:
            h, r, t = triple
            out_str += f'{h}\t{r}\t{t}\n'
    
    with open(out_file, 'w') as f:
        f.write(out_str)


        
