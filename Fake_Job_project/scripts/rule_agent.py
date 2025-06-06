import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import sys


text_model = joblib.load("path-to-model/Fake_Job/model/job_text_model.pkl")
vectorizer = joblib.load("path-to-model/Fake_Job/model/tfidf_vectorizer.pkl")
meta_model = joblib.load("path-to-model/Fake_Job/model/job_meta_model.pkl")

def predict_text_model(text_input: str) -> int:
    vec = vectorizer.transform([text_input])
    return int(text_model.predict(vec)[0])

def predict_meta_model(meta_input: dict) -> int:
    df = pd.DataFrame([meta_input])
    return int(meta_model.predict(df)[0])

def combine_predictions(text_pred: int, meta_pred: int) -> int:
    return text_pred if text_pred == meta_pred else meta_pred

def explain_decision(text_pred: int, meta_pred: int, meta_input: dict) -> str:
    if text_pred == meta_pred:
        return "Both models agree: This job post is {}.".format("REAL" if text_pred else "FAKE")
    else:
        issues = []
        if meta_input["verified_post"] == 0:
            issues.append("unverified")
        if meta_input["view_count"] < 100:
            issues.append("low views")
        if meta_input["days_to_deadline"] < 7:
            issues.append("short deadline")
        reason = ", ".join(issues) if issues else "mixed signals"
        return f"Models disagree. Based on {reason}, this post is likely FAKE."

# === User Input Section ===
print("Enter job details for fraud detection:\n")
lines = []
while True:
    try:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    except EOFError:
        break

text_input = "\n".join(lines)

try:
    days_to_deadline = int(input("Days to deadline (e.g., 10): "))
    verified_post = int(input("Is it verified? (1 = Yes, 0 = No): "))
    view_count = int(input("View count (e.g., 150): "))
except ValueError:
    print("Invalid input. Please enter numeric values.")
    sys.exit()

meta_input = {
    "days_to_deadline": days_to_deadline,
    "verified_post": verified_post,
    "view_count": view_count
}

text_pred = predict_text_model(text_input)
meta_pred = predict_meta_model(meta_input)
final_decision = combine_predictions(text_pred, meta_pred)
explanation = explain_decision(text_pred, meta_pred, meta_input)

print("\n=== AI Agent Decision ===")
print("Text Model Prediction:", "REAL" if text_pred else "FAKE")
print("Metadata Model Prediction:", "REAL" if meta_pred else "FAKE")
print("Final Decision:", "REAL" if final_decision else "FAKE")
print("Explanation:", explanation)
