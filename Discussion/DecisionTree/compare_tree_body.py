import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("C:/Users/QinQin/Desktop/miniMLproject/Discussion/DecisionTree/data/synthetic_body_data.csv")

le = LabelEncoder()
df['BodyShape'] = le.fit_transform(df['BodyShape'])

X = df.drop("BodyShape", axis=1)
y = df["BodyShape"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "CART": DecisionTreeClassifier(criterion="gini"),
    "ID3 (entropy)": DecisionTreeClassifier(criterion="entropy"),
    "C4.5 (entropy/deep)": DecisionTreeClassifier(criterion="entropy", max_depth=None),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Bagging Tree": BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
}

results = []

for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    duration = round(end - start, 3)
    results.append({
        "Model": name,
        "Accuracy (%)": round(accuracy * 100, 2),
        "Training Time (s)": duration
    })

results_df = pd.DataFrame(results).sort_values(by="Accuracy (%)", ascending=False)
print("\nModel Performance Comparison Body Data:")
print(results_df.to_string(index=False))

