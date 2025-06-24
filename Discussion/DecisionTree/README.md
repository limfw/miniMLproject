# Decision Tree Model Design — Does the “Best” Algorithm Always Win?

## Overview

This experiment investigates whether choosing the most complex or widely used machine learning algorithm always leads to the best performance. Using the same ML pipeline, we compare multiple tree-based models on two different classification tasks:

- **Heart Disease Prediction**
- **Body Shape Recommendation**

The key objective is to understand how **data structure, model design, and performance trade-offs** impact the final results — beyond just picking the "best" algorithm.

---

## Datasets

1. `synthetic_body_data.csv`  
 → Simulated physical measurement data for body shape classification

2. `heart_disease_data.csv`  
 → Preprocessed medical dataset for heart disease detection

---

## Models Compared

| Model Type        | Algorithm                                  |
|-------------------|---------------------------------------------|
| Classic Trees     | CART (Gini), ID3 (Entropy), C4.5 (Entropy) |
| Ensemble Methods  | Random Forest, Bagging Tree, XGBoost       |

---

## Results Summary

### Heart Disease Dataset

| Model            | Accuracy (%) | Training Time (s) |
|------------------|--------------|-------------------|
| Bagging Tree     | 88.33        | 0.518             |
| Random Forest    | 86.67        | 0.328             |
| XGBoost          | 86.67        | 0.140             |
| C4.5             | 80.00        | 0.006             |
| ID3              | 75.00        | 0.004             |
| CART             | 75.00        | 0.007             |

### Body Shape Dataset

| Model            | Accuracy (%) | Training Time (s) |
|------------------|--------------|-------------------|
| CART             | 93.10        | 0.009             |
| ID3              | 91.38        | 0.006             |
| C4.5             | 91.38        | 0.007             |
| Random Forest    | 89.66        | 0.420             |
| Bagging Tree     | 87.93        | 0.517             |
| XGBoost          | 87.93        | 0.315             |

---

## Insights

- The best-performing algorithm varies across datasets.
- **Simpler models** like CART outperformed ensembles in the body shape task.
- **Ensemble methods** were more effective for the heart disease prediction.
- **XGBoost**, while powerful, was not always the top performer.

**Conclusion**: Model complexity does not guarantee superior results. Careful **model design, data understanding, and problem framing** are key.

---

## Folder Structure

```

DecisionTree/
│
├── data/
│   ├── synthetic_body_data.csv
│   └── heart_disease_data.csv
│
├── compare_tree_body.py  
├── compare_tree_model_heart.py
└── README.md                 

````
---

## Note
```
This project is part of a discussion series on machine learning model design.

