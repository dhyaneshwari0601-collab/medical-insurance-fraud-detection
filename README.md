# Medical Insurance Fraud Detection Engine

### Anomaly Detection + Graph Machine Learning + NLP Pitch Analysis

A final-year academic project that detects potentially fraudulent medical insurance claims by combining three complementary detection signals into a single Streamlit dashboard.

| Signal | Technique | File |
|---|---|---|
| Statistical anomalies | Isolation Forest | `utils/anomaly_detection.py` |
| Fraud-ring structure | Bipartite Provider-Patient Graph + graph features | `utils/graph_features.py` |
| Claim narrative | TF-IDF + Logistic Regression + lexicon + VADER sentiment | `utils/nlp_pitch_analysis.py` |
| Final fraud prediction | Random Forest Classifier | `utils/fraud_model.py` |

---

## 1. Setup

### Requirements

- Python 3.x
- VS Code or another Python IDE
- Required Python packages listed in `requirements.txt`

### Run the project

```bash
# 1. Open the project folder in VS Code

# 2. Activate the existing virtual environment if available
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate the synthetic dataset
python data/generate_dataset.py

# 5. Run the Streamlit application
streamlit run app.py