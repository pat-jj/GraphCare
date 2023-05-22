from graphcare import *
import random
import pickle

dataset = "mimic3"
kg = "GPT-KG"
tasks = [
    # "mortality", 
    "readmission", 
    "lenofstay", 
    "drugrec"
    ]
ratios = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]


for task in tasks:
    for ratio  in ratios:
        sample_dataset, G, ent2id, rel2id, ent_emb, rel_emb, \
                    map_cluster, map_cluster_inv, map_cluster_rel, map_cluster_rel_inv, \
                        ccscm_id2clus, ccsproc_id2clus, atc3_id2clus = load_everything(dataset, task, kg, kg_ratio=1.0)

        mode, out_channels, loss_function = get_mode_and_out_channels_and_loss_func(task=task, sample_dataset=sample_dataset)

        # label direct ehr node
        print("Labeling direct ehr nodes...")
        sample_dataset = label_ehr_nodes(task, sample_dataset, len(map_cluster), ccscm_id2clus, ccsproc_id2clus, atc3_id2clus)
        # sample_dataset = label_k_hop_nodes(G=G_tg, dataset=sample_dataset, k=2)
        print("Splitting dataset...")
        # train_dataset, val_dataset, test_dataset = split_by_patient(sample_dataset, [0.8, 0.1, 0.1], train_ratio=1.0, seed=528)

        if task == "mortality" or task == "readmission":
            ehr_nodes = set(ccscm_id2clus.values()).union(set(ccsproc_id2clus.values())).union(set(atc3_id2clus.values()))
        else:
            ehr_nodes = set(ccscm_id2clus.values()).union(set(ccsproc_id2clus.values()))

        node_frac = 1- ratio

        nodes = list(G.nodes())
        random.shuffle(nodes)

        num_nodes_to_remove = int(node_frac * G.number_of_nodes())
        nodes_to_remove = nodes[:num_nodes_to_remove]

        for node in nodes_to_remove:
            if str(node) not in ehr_nodes:
                G.remove_node(node)
        G_tg = from_networkx(G)

        for patient in tqdm(sample_dataset):
            node_list = []
            for node in patient['node_set']:
                if node not in nodes_to_remove or str(node) in ehr_nodes:
                    node_list.append(node)

            patient['node_set'] = node_list

        if task == "mortality" or task == "readmission":
            with open(f'../../../data/pj20/exp_data/ccscm_ccsproc_atc3/sample_dataset_mimic3_{task}_th015_kg{ratio}.pkl', 'wb') as f:
                pickle.dump(sample_dataset, f)
        else:
            with open(f'../../../data/pj20/exp_data/ccscm_ccsproc/sample_dataset_mimic3_{task}_th015_kg{ratio}.pkl', 'wb') as f:
                pickle.dump(sample_dataset, f)

