from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.models import RNN
from task_fn import readmission_prediction_mimic3_fn, drug_recommendation_fn, length_of_stay_prediction_mimic3_fn, mortality_prediction_mimic3_fn
from pyhealth.trainer import Trainer
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# STEP 1: load data
base_dataset = MIMIC3Dataset(
    root="/shared/eng/pj20/mimiciii/1.4",
    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    code_mapping={"ICD9CM": "CCSCM", "ICD9PROC": "CCSPROC", "NDC": "ATC"},
    dev=False,
    refresh_cache=False,
)
base_dataset.stat()

# STEP 2: set task
sample_dataset = base_dataset.set_task(mortality_prediction_mimic3_fn)
sample_dataset.stat()
print(sample_dataset.samples[0])

train_dataset, val_dataset, test_dataset = split_by_patient(
    sample_dataset, [0.8, 0.1, 0.1], seed=528
)
train_dataloader = get_dataloader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = get_dataloader(val_dataset, batch_size=4, shuffle=False)
test_dataloader = get_dataloader(test_dataset, batch_size=4, shuffle=False)

# STEP 3: define model
model = RNN(
    dataset=sample_dataset,
    feature_keys=["conditions", "procedures", "drugs"],
    label_key="label",
    mode="binary",
    embedding_dim=128,
)

# STEP 4: define trainer
trainer = Trainer(model=model, metrics=['pr_auc', 'roc_auc'])
trainer.train(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    epochs=50,
    monitor="roc_auc",
    optimizer_params = {"lr": 1e-4},
)

# STEP 5: evaluate
print(trainer.evaluate(test_dataloader))
