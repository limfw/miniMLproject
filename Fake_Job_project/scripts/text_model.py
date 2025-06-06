import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("path-to-data-/data/Hf_fake_data.csv")

# Combine text fields into a single string
df['full_text'] = df[['title', 'company_profile', 'description', 'requirements', 'benefits']].fillna('').agg(' '.join, axis=1)

# Features and target
X = df['full_text']
y = df['fraudulent']

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("=== Text Model Evaluation ===")
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, "path-to-model/model/job_text_model.pkl")
joblib.dump(vectorizer, "path-to-model/model/tfidf_vectorizer.pkl")
print("Text model and vectorizer saved.")
