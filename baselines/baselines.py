from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset
from pyhealth.models import Deepr, ConCare, AdaCare, GRASP, StageNet, MoleRec, SafeDrug, GAMENet, RNN, Transformer, RETAIN, MICRON
from pyhealth.tasks import drug_recommendation_mimic3_fn, mortality_prediction_mimic3_fn, readmission_prediction_mimic3_fn, length_of_stay_prediction_mimic3_fn, mortality_prediction_mimic4_fn, readmission_prediction_mimic4_fn, length_of_stay_prediction_mimic4_fn, drug_recommendation_mimic4_fn
from pyhealth.trainer import Trainer
import pickle
from pyhealth.datasets import split_by_patient
from pyhealth.datasets import get_dataloader
import torch
import json
from collections import defaultdict

datasets = [
    "mimic3",
    # "mimic4",
]


tasks = [
    # "mortality",
    # "readmission",
    # "lenofstay",
    "drugrec"
]

train_ratios = \
[
    0.001,
    0.002,
    0.003,
    0.004,
    0.005,
    0.006,
    0.007,
    0.008,
    0.009,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.1,
    0.3,
    0.5,
    0.7,
    0.9,
    1.0,
]

feat_ratios = [
    # 0.05,
    # 0.1,
    # 0.2,
    # 0.3,
    # 0.4,
    # 0.5,
    # 0.6,
    # 0.7,
    # 0.8,
    # 0.9,
]



device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

for dataset in datasets:
    if dataset == "mimic3":
        base_dataset = MIMIC3Dataset(
            root="/data/physionet.org/files/mimiciii/1.4/",
            tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
            code_mapping={
                "NDC": ("ATC", {"target_kwargs": {"level": 3}}),
                "ICD9CM": "CCSCM",
                "ICD9PROC": "CCSPROC"
                },
            dev=False,
            refresh_cache=False,
        )
    elif dataset == "mimic4":
        base_dataset = MIMIC4Dataset(
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

        
    for task in tasks:
        if task == "mortality":
            if dataset == "mimic4":
                sample_dataset = base_dataset.set_task(mortality_prediction_mimic4_fn)
            else:
                sample_dataset = base_dataset.set_task(mortality_prediction_mimic3_fn)

        elif task == "readmission":
            if dataset == "mimic4":
                sample_dataset = base_dataset.set_task(readmission_prediction_mimic4_fn)
            else:
                sample_dataset = base_dataset.set_task(readmission_prediction_mimic3_fn)
        elif task == "lenofstay":
            if dataset == "mimic4":
                sample_dataset = base_dataset.set_task(length_of_stay_prediction_mimic4_fn)
            else:
                sample_dataset = base_dataset.set_task(length_of_stay_prediction_mimic3_fn)
        elif task == "drugrec":
            if dataset == "mimic4":
                sample_dataset = base_dataset.set_task(drug_recommendation_mimic4_fn)
            else:    
                sample_dataset = base_dataset.set_task(drug_recommendation_mimic3_fn)

        if task == "mortality" or task == "readmission":
            models = [
                RNN,
                Transformer,
                RETAIN,
                Deepr,
                AdaCare,
                GRASP,
                StageNet,
            ]
        elif task == "lenofstay":
            models = [
                RNN,
                Transformer,
                RETAIN,
                Deepr,
                StageNet,
            ]
        elif task == "drugrec":
            models = [
                # RNN,
                # Deepr,
                StageNet
                # Transformer,
                # RETAIN,
                # SafeDrug,
                # MICRON,
                # GAMENet,
                # MoleRec,
            ]
        for train_ratio in train_ratios:

                results = defaultdict(list)
                for j in range(3):

                    train_dataset, val_dataset, test_dataset = split_by_patient(sample_dataset, [0.8, 0.1, 0.1], train_ratio=train_ratio, seed=528)
                    if len(feat_ratios) == 0:
                        train_loader = get_dataloader(train_dataset, batch_size=64, shuffle=True)
                    # else:
                    #     with open(f'/data/pj20/exp_data/ccscm_ccsproc/train_dataset_mimic3_{task}_th015_{ratio}.pkl', 'rb') as f:
                    #         train_dataset = pickle.load(f)
                    #         train_loader = get_dataloader(train_dataset, batch_size=64, shuffle=True)

                    val_loader = get_dataloader(val_dataset, batch_size=64, shuffle=False)
                    # if len(feat_ratios) == 0:
                    #     val_loader = get_dataloader(val_dataset, batch_size=64, shuffle=False)
                    # else:
                    #     loaders = []
                    #     for ratio in feat_ratios:
                    #         with open(f'/data/pj20/exp_data/ccscm_ccsproc/val_dataset_mimic3_{task}_th015_{ratio}.pkl', 'rb') as f:
                    #             val_dataset = pickle.load(f)
                    #             val_loader = get_dataloader(val_dataset, batch_size=64, shuffle=False)
                    #             loaders.append(val_loader)
                    

                    test_loader = get_dataloader(test_dataset, batch_size=64, shuffle=False)
                    print("train_ratio: ", train_ratio)

                    for i in range(len(models)):
                        if task == "mortality" or task == "readmission":
                            if i != 4 and i != 5:
                                model = models[i](
                                    dataset=sample_dataset,
                                    feature_keys=["conditions", "procedures", "drugs"],
                                    label_key="label",
                                    mode="binary",
                                )
                            else:
                                model = models[i](
                                    dataset=sample_dataset,
                                    feature_keys=["conditions", "procedures", "drugs"],
                                    label_key="label",
                                    use_embedding=[True, True, True],
                                    mode="binary",
                                )
                                

                            trainer = Trainer(model=model, device=device, metrics=["pr_auc", "roc_auc", "accuracy", "f1", "jaccard"])
                            trainer.train(
                                train_dataloader=train_loader,
                                epochs=50,
                                monitor="roc_auc",
                            )

                        elif task == "lenofstay":
                            
                            try:
                                model = models[i](
                                    dataset=sample_dataset,
                                    feature_keys=["conditions", "procedures"],
                                    label_key="label",
                                    mode="multiclass",
                                )


                                ## multi-class
                                trainer = Trainer(model=model, device=device, metrics=["roc_auc_weighted_ovr", "cohen_kappa", "accuracy", "f1_weighted"])
                                trainer.train(
                                    train_dataloader=train_loader,
                                    epochs=50,
                                    monitor="roc_auc_weighted_ovr",
                                )
                            except:
                                continue

                        elif task == "drugrec":
                            # try:
                            model = models[i](
                                dataset=sample_dataset,
                                feature_keys=["conditions", "procedures"],
                                label_key="drugs",
                                mode="multilabel",
                            )
                            # except:
                            # model = models[i](dataset=sample_dataset)

                                    ## multi-label
                            trainer = Trainer(model=model, device=device, metrics=["pr_auc_samples", "roc_auc_samples", "f1_samples", "jaccard_samples"])
                            # try:
                            trainer.train(
                                train_dataloader=train_loader,
                                epochs=50,
                                monitor="pr_auc_samples",
                            )

                            # except:
                            #     try:
                            results[models[i].__name__].append(trainer.evaluate(val_loader))
                                # except:
                                #     continue
                        
                        # try:
                        #     results[models[i].__name__].append(trainer.evaluate(val_loader))
                        # except:
                        #     continue

                # processed_results = defaultdict(dict)
                # for i in range(len(feat_ratios)):
                #     for k, v in results.items():
                #         for kk, vv in v[i].items():
                #             processed_results[k][kk] = defaultdict(list)
                
                # for i in range(len(feat_ratios)):
                #     for k, v in results.items():
                #         for kk, vv in v[i].items():
                #             processed_results[k][kk].append(vv)


                # with open(f"./output/results_{dataset}_{task}_feat_{ratio}_new.json", "w") as f:
                #     json.dump(results, f, indent=6)


                avg_results = defaultdict(dict)

                for k, v in results.items():
                    for k_, v_ in v[0].items():
                        avg_results[k][k_] = sum([vv[k_] for vv in v]) / len(v)


                import numpy as np
                # calculate standard deviation
                variation_results = defaultdict(dict)

                for k, v in results.items():
                    for k_, v_ in v[0].items():
                        variation_results[k][k_] = np.std([vv[k_] for vv in v])


                print(avg_results)
                print(variation_results)
                with open(f"./output/avg_results_{dataset}_{task}_{train_ratio}_new.json", "w") as f:
                    json.dump(avg_results, f, indent=6)
                with open(f"./output/variation_results_{dataset}_{task}_{train_ratio}_new.json", "w") as f:
                    json.dump(variation_results, f, indent=6)
            


    




