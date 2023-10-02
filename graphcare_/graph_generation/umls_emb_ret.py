from get_emb import embedding_retriever
import numpy as np
from tqdm import tqdm
import pickle
from concurrent.futures import ThreadPoolExecutor

SAVE_INTERVAL = 10000  # Save after processing every 1000 names
MAX_RETRIES = 30  # Retry up to 5 times if there's an error

# Load previous embeddings if they exist
try:
    with open('/data/pj20/exp_data/umls_ent_emb_.pkl', 'rb') as f:
        umls_ent_emb = pickle.load(f)
except FileNotFoundError:
    umls_ent_emb = []

# Loading and preprocessing the names
with open("/home/pj20/GraphCare/KG_mapping/umls/concept_names.txt", 'r') as f:
    umls_ent = f.readlines()

umls_names = [line.split('\t')[1][:-1] for line in umls_ent]

# Skip names that are already processed
umls_names = umls_names[len(umls_ent_emb):]

def get_embedding(name):
    for _ in range(MAX_RETRIES):
        try:
            emb = embedding_retriever(term=name)
            return emb
        except KeyError:
            pass  # Retry on KeyError
    return "Error: Failed to retrieve embedding for {}".format(name)

# Use ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=20) as executor:
    for idx, emb in enumerate(tqdm(executor.map(get_embedding, umls_names), total=len(umls_names))):
        umls_ent_emb.append(emb)
        
        # Periodically save the data
        if (idx + 1) % SAVE_INTERVAL == 0:
            with open('/data/pj20/exp_data/umls_ent_emb_.pkl', 'wb') as f:
                pickle.dump(umls_ent_emb, f)

# Save the final data
with open('/data/pj20/exp_data/umls_ent_emb_.pkl', 'wb') as f:
    pickle.dump(umls_ent_emb, f)
