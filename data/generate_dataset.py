"""
generate_dataset.py
--------------------
Generates a synthetic Medical Insurance Claims dataset that mirrors the schema
and statistical patterns of the public Kaggle dataset:

    "Healthcare Provider Fraud Detection Analysis"
    https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis

WHY SYNTHETIC DATA IS USED
---------------------------
This environment cannot reach kaggle.com to download the dataset directly
(no internet egress to Kaggle is permitted here). To keep the project fully
runnable out-of-the-box, this script generates a realistic stand-in dataset
with the SAME COLUMN STRUCTURE, fraud-ring patterns, and claim-narrative
("pitch") text that the real pipeline expects.

TO USE THE REAL KAGGLE DATA INSTEAD:
1. Download "Train-1542865627584.csv", "Train_Beneficiarydata-1542865627584.csv",
   and "Train_Inpatientdata-1542865627584.csv" / Outpatient file from Kaggle.
2. Merge them on Provider / BeneID exactly like the notebook in the Kaggle
   discussion boards does, producing one row per claim with a `PotentialFraud`
   label.
3. Rename columns to match `data/claims.csv` produced here (see README.md,
   section "Swapping in the real Kaggle dataset") and drop this generator.

The rest of the pipeline (anomaly detection, graph features, NLP pitch
analysis, fraud model) is dataset-agnostic and needs no changes.
"""

import numpy as np
import pandas as pd
from faker import Faker
import random

fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

N_PROVIDERS = 250
N_PATIENTS = 1400
N_CLAIMS = 6000
FRAUD_PROVIDER_RATE = 0.10          # 10% of providers are fraud rings
DIAGNOSIS_CODES = [f"ICD{i:03d}" for i in range(1, 60)]
PROCEDURE_CODES = [f"CPT{i:04d}" for i in range(1000, 1080)]

GENUINE_PHRASES = [
    "Patient presented with {symptom}. Standard workup ordered including {test}.",
    "Follow-up visit for {symptom}, condition stable, continuing current treatment plan.",
    "Routine {test} performed as part of chronic disease management for {symptom}.",
    "Patient referred by primary care for evaluation of {symptom}; {test} recommended.",
    "Post-operative check following prior treatment, no complications noted.",
    "Annual wellness exam, patient reports mild {symptom}, monitoring advised.",
]

SUSPICIOUS_PHRASES = [
    "Emergency treatment required immediately, please approve claim with no delay, guaranteed reimbursement.",
    "Patient requested cash settlement, multiple procedures billed same day, no additional documentation needed.",
    "Urgent approval needed, paperwork will follow later, trust the provider on this one.",
    "Bulk procedures performed for convenience, all codes billed at maximum allowable rate.",
    "Claim resubmitted under different code after previous denial, please process quickly.",
    "Provider guarantees full recovery, extensive imaging and tests billed as precaution only.",
]

SYMPTOMS = ["chest discomfort", "lower back pain", "chronic fatigue", "joint stiffness",
            "shortness of breath", "abdominal pain", "recurring headaches", "mild fever"]
TESTS = ["blood panel", "MRI scan", "X-ray", "ECG", "CT scan", "ultrasound", "biopsy"]


def make_narrative(is_fraud_claim: bool) -> str:
    if is_fraud_claim and random.random() < 0.55:
        return random.choice(SUSPICIOUS_PHRASES)
    template = random.choice(GENUINE_PHRASES)
    return template.format(symptom=random.choice(SYMPTOMS), test=random.choice(TESTS))


def generate():
    # --- Providers -----------------------------------------------------
    provider_ids = [f"PRV{i:05d}" for i in range(N_PROVIDERS)]
    fraud_providers = set(random.sample(provider_ids,
                                         int(N_PROVIDERS * FRAUD_PROVIDER_RATE)))

    # --- Patients --------------------------------------------------------
    patient_ids = [f"PAT{i:06d}" for i in range(N_PATIENTS)]
    patient_age = {p: np.random.randint(18, 90) for p in patient_ids}
    patient_gender = {p: random.choice(["M", "F"]) for p in patient_ids}
    patient_chronic = {p: np.random.poisson(1.5) for p in patient_ids}

    # Fraud rings: a small clique of providers shares a tight cluster of
    # "recycled" patients across many claims -> this is exactly the kind
    # of dense-subgraph signal graph features are built to catch.
    fraud_patient_pool = random.sample(patient_ids, int(N_PATIENTS * 0.12))

    rows = []
    for i in range(N_CLAIMS):
        provider = random.choice(provider_ids)
        is_fraud_provider = provider in fraud_providers

        if is_fraud_provider and random.random() < 0.75:
            patient = random.choice(fraud_patient_pool)
        else:
            patient = random.choice(patient_ids)

        base_amount = np.random.gamma(shape=2.0, scale=250)
        if is_fraud_provider:
            # inflated / erratic billing is a classic fraud signature
            amount = base_amount * np.random.uniform(2.5, 6.0)
        else:
            amount = base_amount

        claim_start = fake.date_between(start_date="-2y", end_date="-1y")
        claim_len = np.random.poisson(2) if not is_fraud_provider else np.random.poisson(1)
        claim_end = pd.to_datetime(claim_start) + pd.Timedelta(days=int(claim_len))

        label = 1 if (is_fraud_provider and random.random() < 0.65) else 0

        rows.append({
            "ClaimID": f"CLM{i:07d}",
            "ProviderID": provider,
            "PatientID": patient,
            "PatientAge": patient_age[patient],
            "PatientGender": patient_gender[patient],
            "ChronicConditions": patient_chronic[patient],
            "ClaimStartDt": claim_start,
            "ClaimEndDt": claim_end.date(),
            "ClaimDurationDays": claim_len,
            "ClaimAmount": round(float(amount), 2),
            "DiagnosisCode": random.choice(DIAGNOSIS_CODES),
            "ProcedureCode": random.choice(PROCEDURE_CODES),
            "NumProceduresBilled": np.random.poisson(3 if is_fraud_provider else 1) + 1,
            "ClaimNarrative": make_narrative(is_fraud_provider),
            "PotentialFraud": label,
        })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/claims.csv", index=False)
    print(f"Generated {len(df)} claims -> data/claims.csv")
    print(f"Fraud rate: {df['PotentialFraud'].mean():.2%}")
