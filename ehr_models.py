from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset
from graphcare_.task_fn import drug_recommendation_fn, drug_recommendation_mimic4_fn, mortality_prediction_mimic3_fn, readmission_prediction_mimic3_fn, length_of_stay_prediction_mimic3_fn, length_of_stay_prediction_mimic4_fn, mortality_prediction_mimic4_fn, readmission_prediction_mimic4_fn
from pyhealth.datasets import get_dataloader
from graphcare_ import split_by_patient
import pickle
from pyhealth.trainer import Trainer
import torch
from pyhealth.models import Transformer, RETAIN, SafeDrug, MICRON, CNN, RNN, GAMENet
from collections import defaultdict
import json

tasks = \
[
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
]

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

for task in tasks:
    print("task: ", task)
    if task == "mortality" or task == "readmission":
        with open(f'../../../data/pj20/exp_data/ccscm_ccsproc_atc3/sample_dataset_mimic3_{task}_th015.pkl', 'rb') as f:
            sample_dataset = pickle.load(f)
    else:
        with open(f'../../../data/pj20/exp_data/ccscm_ccsproc/sample_dataset_mimic3_{task}_th015.pkl', 'rb') as f:
            sample_dataset = pickle.load(f)
    for train_ratio in train_ratios:

        if task != "drugrec":
            models = [RNN, Transformer, RETAIN]
        else:
            models = [
                Transformer, 
                RETAIN, 
                # SafeDrug, 
                MICRON, 
                # GAMENet
                ]


        results = defaultdict(list)

        for i in range(5):
            print("train_ratio: ", train_ratio)
            train_dataset, val_dataset, test_dataset = split_by_patient(sample_dataset, [0.8, 0.1, 0.1], train_ratio=train_ratio, seed=528)
            train_loader = get_dataloader(train_dataset, batch_size=64, shuffle=True)
            val_loader = get_dataloader(val_dataset, batch_size=64, shuffle=False)
            test_loader = get_dataloader(test_dataset, batch_size=64, shuffle=False)
            for model_ in models:
                if task == "mortality" or task == "readmission":
                    model = model_(
                        dataset=sample_dataset,
                        feature_keys=["conditions", "procedures", "drugs"],
                        label_key="label",
                        mode="binary",
                    )
                    ## binary
                    trainer = Trainer(model=model, device=device, metrics=["pr_auc", "roc_auc", "accuracy", "f1", "jaccard"])
                    trainer.train(
                        train_dataloader=train_loader,
                        val_dataloader=val_loader,
                        epochs=5,
                        monitor="accuracy",
                    )

                elif task == "lenofstay":
                    model = model_(
                        dataset=sample_dataset,
                        feature_keys=["conditions", "procedures"],
                        label_key="label",
                        mode="multiclass",
                    )

                    ## multi-class
                    trainer = Trainer(model=model, device=device, metrics=["roc_auc_weighted_ovr", "cohen_kappa", "accuracy", "f1_weighted"])
                    trainer.train(
                        train_dataloader=train_loader,
                        val_dataloader=val_loader,
                        epochs=5,
                        monitor="roc_auc_weighted_ovr",
                    )

                elif task == "drugrec":
                    try:
                        model = model_(
                            dataset=sample_dataset,
                            feature_keys=["conditions", "procedures"],
                            label_key="drugs",
                            mode="multilabel",
                        )
                    except:
                        model = model_(dataset=sample_dataset)

                            ## multi-label
                    trainer = Trainer(model=model, device=device, metrics=["pr_auc_samples", "roc_auc_samples", "f1_samples", "jaccard_samples"])
                    try:
                        trainer.train(
                            train_dataloader=train_loader,
                            val_dataloader=val_loader,
                            epochs=5,
                            monitor="pr_auc_samples",
                        )

                    except:
                        try:
                            results[model_.__name__].append(trainer.evaluate(val_loader))
                        except:
                            continue
                        continue

                results[model_.__name__].append(trainer.evaluate(val_loader))


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
        with open(f"./ehr_training_result/avg_results_{task}_{train_ratio}.json", "w") as f:
            json.dump(avg_results, f, indent=6)
        with open(f"./ehr_training_result/variation_results_{task}_{train_ratio}.json", "w") as f:
            json.dump(variation_results, f, indent=6)
