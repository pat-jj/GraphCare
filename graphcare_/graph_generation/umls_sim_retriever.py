import pickle
import numpy as np
from tqdm import tqdm
import multiprocessing

def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def find_most_similar_embedding(target_emb, umls_ent_emb):
    max_similarity = -1
    max_index = None
    for idx, umls_emb in enumerate(umls_ent_emb):
        similarity = cosine_similarity(target_emb, umls_emb)
        if similarity > max_similarity:
            max_similarity = similarity
            max_index = idx
    if max_similarity > SIMILARITY_THRESHOLD:
        return max_index
    else:
        return None

def process_chunk(chunk, id2emb, umls_ent_emb, output_dict):
    for key in chunk:
        output_dict[key] = find_most_similar_embedding(id2emb[key], umls_ent_emb)

def parallel_mapping(id2emb, umls_ent_emb, num_processes):
    keys = list(id2emb.keys())
    chunks = np.array_split(keys, num_processes)
    manager = multiprocessing.Manager()
    output_dict = manager.dict()
    processes = []

    for chunk in chunks:
        p = multiprocessing.Process(target=process_chunk, args=(chunk, id2emb, umls_ent_emb, output_dict))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    return dict(output_dict)

if __name__ == "__main__":
    with open("/home/pj20/GraphCare/KG_mapping/umls/concept_names.txt", 'r') as f:
        umls_ent = f.readlines() 
    
    umls_ids = []
    umls_names = []

    for line in umls_ent:
        umls_id = line.split('\t')[0]
        umls_name = line.split('\t')[1][:-1]
        umls_names.append(umls_name)
        umls_ids.append(umls_id)
        
        

    with open('/data/pj20/exp_data/umls_ent_emb_.pkl', 'rb') as f:
        umls_ent_emb = pickle.load(f)
        
    with open('/data/pj20/exp_data/atc3_id2emb.pkl', 'rb') as f:
        atc3_id2emb = pickle.load(f)
        
    with open('/data/pj20/exp_data/ccscm_id2emb.pkl', 'rb') as f:
        ccscm_id2emb = pickle.load(f)
        
    with open('/data/pj20/exp_data/ccsproc_id2emb.pkl', 'rb') as f:
        ccsproc_id2emb = pickle.load(f)



    ccscm2umls, ccsproc2umls, atc32umls = {}, {}, {}

    # Define a similarity threshold
    SIMILARITY_THRESHOLD = 0.7
    num_processes = 30

    ccscm2umls = parallel_mapping(ccscm_id2emb, umls_ent_emb, num_processes)
    ccsproc2umls = parallel_mapping(ccsproc_id2emb, umls_ent_emb, num_processes)
    atc32umls = parallel_mapping(atc3_id2emb, umls_ent_emb, num_processes)

    
    
    with open('/data/pj20/exp_data/ccscm2umls.pkl', 'wb') as f:
        pickle.dump(ccscm2umls, f)
        
    with open('/data/pj20/exp_data/ccsproc2umls.pkl', 'wb') as f:
        pickle.dump(ccsproc2umls, f)
        
    with open('/data/pj20/exp_data/atc32umls.pkl', 'wb') as f:
        pickle.dump(atc32umls, f)
