import csv

condition_mapping_file = "../../resources/CCSCM.csv"
procedure_mapping_file = "../../resources/CCSPROC.csv"
drug_file = "../../resources/ATC.csv"

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


from tqdm import tqdm
import json

drug_ent = set()
drug_rel = set()

file_dir = "../../graphs/drug/ATC3"

for key in drug_dict.keys():
    file = f"{file_dir}/{key}.txt"
    with open(file=file, mode='r') as f:
        lines = f.readlines()
    
    for line in lines:
        parsed = line.split('\t')
        if len(parsed) == 3:
            h, r, t = line.split('\t')
            t = t[:-1]
            drug_ent.add(h)
            drug_ent.add(t)
            drug_rel.add(r)


drug_id2ent = {index: value for index, value in enumerate(drug_ent)}
drug_ent2id = {value: index for index, value in enumerate(drug_ent)}
drug_id2rel = {index: value for index, value in enumerate(drug_rel)}
drug_rel2id = {value: index for index, value in enumerate(drug_rel)}

out_file_id2ent = f"{file_dir}/id2ent.json"
out_file_ent2id = f"{file_dir}/ent2id.json"
out_file_id2rel = f"{file_dir}/id2rel.json"
out_file_rel2id = f"{file_dir}/rel2id.json"

with open(out_file_id2ent, 'w') as file:
    json.dump(drug_id2ent, file, indent=6)
with open(out_file_ent2id, 'w') as file:
    json.dump(drug_ent2id, file, indent=6)
with open(out_file_id2rel, 'w') as file:
    json.dump(drug_id2rel, file, indent=6)
with open(out_file_rel2id, 'w') as file:
    json.dump(drug_rel2id, file, indent=6)
    

import json

file_dir = "../../graphs/drug/ATC3"

file_id2ent = f"{file_dir}/id2ent.json"
file_ent2id = f"{file_dir}/ent2id.json"
file_id2rel = f"{file_dir}/id2rel.json"
file_rel2id = f"{file_dir}/rel2id.json"

with open(file_id2ent, 'r') as file:
    drug_id2ent = json.load(file)
with open(file_ent2id, 'r') as file:
    drug_ent2id = json.load(file)
with open(file_id2rel, 'r') as file:
    drug_id2rel = json.load(file)
with open(file_rel2id, 'r') as file:
    drug_rel2id = json.load(file)


from get_emb import embedding_retriever
import numpy as np
from tqdm import tqdm
import pickle

# get embedding for drug entities
drug_ent_emb = []

for idx in tqdm(range(len(drug_id2ent))):
    ent = drug_id2ent[str(idx)]
    try:
        embedding = embedding_retriever(term=ent)
        embedding = np.array(embedding)
    except:
        embedding = np.random.randn(1536)

    drug_ent_emb.append(embedding)

stacked_embedding = np.vstack(drug_ent_emb)
emb_pkl = f"{file_dir}/entity_embedding.pkl"

with open(emb_pkl, "wb") as file:
    pickle.dump(stacked_embedding, file)

# get embedding for drug relations
drug_rel_emb = []

for idx in tqdm(range(len(drug_id2rel))):
    rel = drug_id2rel[str(idx)]
    try:
        embedding = embedding_retriever(term=rel)
        embedding = np.array(embedding)
    except:
        embedding = np.random.randn(1536)

    drug_rel_emb.append(embedding)

stacked_embedding = np.vstack(drug_rel_emb)

emb_pkl = f"{file_dir}/relation_embedding.pkl"

with open(emb_pkl, "wb") as file:
    pickle.dump(stacked_embedding, file)