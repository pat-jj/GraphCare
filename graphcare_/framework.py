from .text_retrieve import retrieve_text_from_corpus
from .kg_generator import generate_customized_kg
from .kg_embedding import load_pretrained_kg_embedding_model, embed_customized_kg_using_pretrained_model
from .ehr_embedding import generate_ehr_embedding
from .utils import concatenate_embeddings
from .task_fn import drug_recommendation_fn
from .code_mapping import translate
from .model import GraphCare

from pyhealth.datasets import MIMIC3Dataset


def construct_dataset():

    mimic3_ds = MIMIC3Dataset(
        root="../../../data/physionet.org/files/mimiciii/1.4/", 
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],      
        code_mapping={
            "NDC": ("ATC", {"target_kwargs": {"level": 3}}),
            "ICD9CM": "CCSCM",
            "ICD9PROC": "CCSPROC"
            },
    )

    sample_dataset = mimic3_ds.set_task(drug_recommendation_fn)
    sample_dataset_name = translate(sample_dataset)

    return sample_dataset, sample_dataset_name



def run():

    return