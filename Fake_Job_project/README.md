```markdown
#Fake Job Post Detection – Dual-Model Ensemble Project

This project builds a machine learning system to **detect fake job postings** using two different models:
- A **text-based classifier** using job descriptions.
- A **metadata-based classifier** using structured information like salary and employment type.

The outputs are then **combined using a rule-based agent** to produce final predictions and human-readable explanations.

---

## Project Structure

```

Fake-job-project/
├── data/                               # Datasets
│   ├── Job_data.csv                    # Hugging Face dataset
│   └── Job_metadata.csv                # Generated metadata with labels
│
├── model/                              # Saved models
│   ├── Job_meta_model.pkl              # Random Forest on metadata
│   └── job_text_model.pkl              # TF-IDF + Logistic Regression
│   └── tfidf_vectorizer.pkl
│
├── scripts/                            # code
│   ├── text_model.py                   # TF-IDF + Logistic Regression
│   ├── metadata\_model.py              # Random Forest on metadata
│   ├── rule\_agent.py                  # Combine outputs with logic
|
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation

````
---
## Dataset Description
### 1. Hugging Face Dataset  
Source: [`victor/real-or-fake-fake-jobposting-prediction`](https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction)  
- Contains job title, description, company profile, requirements, etc.
- Label: `fraudulent` (0 = real, 1 = fake)

### 2. Synthetic Metadata Dataset  
- Generated to match the Hugging Face labels
- Includes features such as salary, telecommute, has_company_logo, etc.

---

## Model Pipelines

### Case 1: Text-Based Model
- Preprocessing: TF-IDF vectorization on combined text fields
- Model: Logistic Regression
- Output: Binary prediction + probability of fraud

Script: `scripts/text_model.py`

---

### Case 2: Metadata-Based Model
- Preprocessing: Feature engineering on structured fields
- Model: Random Forest Classifier
- Output: Binary prediction + probability of fraud

Script: `scripts/metadata_model.py`

---

### Rule-Based Agent
- Combines predictions from both models with a simple decision rule:

```python
def combine_predictions(text_pred: int, meta_pred: int) -> int:
    if text_pred == 1 and meta_pred == 1:
        return 1
    elif text_pred == 0 and meta_pred == 0:
        return 0
    else:
        return 1  # cautious default: flag as fake if disagreement

def explain_decision(text_pred: int, meta_pred: int, meta_input: dict) -> str:
    if text_pred == 1 and meta_pred == 1:
        return "Both models indicate FAKE."
    elif text_pred == 0 and meta_pred == 0:
        return "Both models indicate REAL."
    else:
        return "Disagreement — flagged as FAKE for caution."

````

📄 Script: `scripts/rule_agent.py`

---

## How to Run

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Train and save models**

```bash
python scripts/text_model.py
python scripts/metadata_model.py
```

3. **Run rule-based agent on new data**

```bash
python scripts/rule_agent.py
```

---

## Notes
* The goal is to demonstrate modular, explainable ML design for fraud detection use cases.

---