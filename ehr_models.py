from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset
from GraphCare.task_fn import drug_recommendation_fn, drug_recommendation_mimic4_fn, mortality_prediction_mimic3_fn, readmission_prediction_mimic3_fn, length_of_stay_prediction_mimic3_fn, length_of_stay_prediction_mimic4_fn, mortality_prediction_mimic4_fn, readmission_prediction_mimic4_fn

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

sample_dataset = ds.set_task(drug_recommendation_mimic4_fn)


# import pickle

# with open('../../../data/pj20/exp_data/ccscm_ccsproc/sample_dataset_mimic4_drugrec_th015.pkl', 'rb') as f:
#     sample_dataset = pickle.load(f)


from pyhealth.datasets import split_by_patient, get_dataloader

train_dataset, val_dataset, test_dataset = split_by_patient(sample_dataset, [0.8, 0.1, 0.1], seed=528)
train_loader = get_dataloader(train_dataset, batch_size=64, shuffle=True)
val_loader = get_dataloader(val_dataset, batch_size=64, shuffle=False)
test_loader = get_dataloader(test_dataset, batch_size=64, shuffle=False)


from pyhealth.trainer import Trainer
import torch
from pyhealth.models import Transformer, RETAIN, SafeDrug, MICRON, CNN, RNN, GAMENet
from collections import defaultdict

results = defaultdict(list)

for i in range(3):
    for model_ in [
        # SafeDrug,
        Transformer, 
        RETAIN,
        MICRON,
        GAMENet
        ]:
        try:
            model = model_(
                dataset=sample_dataset,
                feature_keys=["conditions", "procedures"],
                label_key="drugs",
                mode="multilabel",
            )
        except:
            model = model_(dataset=sample_dataset)

        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        ## binary
        # trainer = Trainer(model=model, device=device, metrics=["pr_auc", "roc_auc", "accuracy", "f1", "jaccard"])
        # trainer.train(
        #     train_dataloader=train_loader,
        #     val_dataloader=val_loader,
        #     epochs=5,
        #     monitor="accuracy",
        # )

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

        ## multi-class
        # trainer = Trainer(model=model, device=device, metrics=["roc_auc_weighted_ovr", "cohen_kappa", "accuracy", "f1_weighted"])
        # trainer.train(
        #     train_dataloader=train_loader,
        #     val_dataloader=val_loader,
        #     epochs=5,
        #     monitor="roc_auc_weighted_ovr",
        # )

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
print(avg_results, file=open("avg_results.txt", "w"))
print(variation_results, file=open("variation_results.txt", "w"))
