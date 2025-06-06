import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datetime import timedelta

# Load original HF dataset
df = pd.read_csv("path-to-data/data/Hf_metadata.csv")

# Select features and target
features = ['days_to_deadline', 'verified_post', 'view_count']
X = df[features]
y = df['fraudulent']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("=== Metadata Model Evaluation ===")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "path-to-model/Fake_Job/model/job_meta_model.pkl")
print("Metadata model saved as job_meta_model.pkl")
