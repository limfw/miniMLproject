# Code complexity : Machine Learning  vs Rule-Based Product Prediction: A Retail Recommendation Demo

This project compares two approaches to predicting which product tier a customer is likely to buy:
- **Rule-Based Logic** (manual IF-ELSE rules)
- **Machine Learning** (Decision Tree classifier)

---

## Problem Overview

In a retail setting, companies often classify customers into product segments such as:
- **Premium**
- **Mid-Range**
- **Budget**

Traditional systems use **hardcoded rules** (e.g., income > 80k → Premium), but this approach struggles when data becomes complex or customer behavior changes. This project shows why **machine learning** is a better fit for such dynamic tasks.

---

## Files Included

| File                | Description                                              |
|---------------------|----------------------------------------------------------|
| `data/syntheticData.csv` | Simulated customer data (features + product labels) |
| `hardcopy.py`       | Rule-based predictor using 8 handcrafted business rules |
| `decision_tree.py`  | ML-based predictor using a Decision Tree                |
| `requirements.txt`  | Python dependencies list                                |

---

## How It Works

### `hardcopy.py` (Rule-Based)

- Uses 8 handcrafted rules based on income, job, loyalty, etc.
- Predicts one of 3 segments: Premium, Mid-Range, or Budget
- Good for simple logic, but hard to scale or adapt

### `decision_tree.py` (Machine Learning)

- Uses scikit-learn to train a Decision Tree Classifier
- Learns optimal rules directly from the data
- Adapts automatically as customer behavior evolves

---

## Note on Fallback Handling

In `hardcopy.py`, if no rule matches a customer, a fallback label ("Mid-Range") is assigned.  
Due to partial condition checks during fallback analysis, you may see a **negative fallback count** — this is not an error in prediction, but an artifact of how fallback is estimated.

---

## What You'll See

- Actual vs Predicted distribution comparison
- Breakdown of which rules were triggered
- Random samples for manual inspection
- ML model visualization of the learned decision tree

---
