import csv


condition_mapping_file = "./resources/CCSCM.csv"
procedure_mapping_file = "./resources/CCSPROC.csv"
drug_file = "./resources/ATC.csv"

condition_dict = {}
with open(condition_mapping_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        condition_dict[row['code']] = row['name']

procedure_dict = {}
with open(procedure_mapping_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        procedure_dict[row['code']] = row['name']

drug_dict = {}
with open(drug_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        condition_dict[row['code']] = row['name']


def translate(data):
    mapped_data = []

    for record in data:
        mapped_record = {}
        mapped_record['visit_id'] = record['visit_id']
        mapped_record['patient_id'] = record['patient_id']

        condition_list = []
        for condition in record['conditions']:
            condition_names = []
            for code in condition:
                if code in condition_dict:
                    condition_names.append(condition_dict[code])
                else:
                    condition_names.append(code)
            condition_list.append(condition_names)
        mapped_record['conditions'] = condition_list

        procedure_list = []
        for procedure in record['procedures']:
            procedure_names = []
            for code in procedure:
                if code in procedure_dict:
                    procedure_names.append(procedure_dict[code])
                else:
                    procedure_names.append(code)
            procedure_list.append(procedure_names)
        mapped_record['procedures'] = procedure_list

        drug_names = []
        for code in record['drugs']:
            if code in drug_dict:
                drug_names.append(drug_dict[code])
            else:
                drug_names.append(code)
        mapped_record['drugs'] = drug_names

        drugs_all_list = []
        for drugs in record['drugs_all']:
            drug_names = []
            for code in drugs:
                if code in drug_dict:
                    drug_names.append(drug_dict[code])
                else:
                    drug_names.append(code)
            drugs_all_list.append(drug_names)
        mapped_record['drugs_all'] = drugs_all_list

        mapped_data.append(mapped_record)

    return mapped_data

