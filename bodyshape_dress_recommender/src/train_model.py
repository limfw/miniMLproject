import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib


current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

data_path = os.path.join(parent_dir, 'data', 'synthetic_body_data.csv')
model_dir = os.path.join(parent_dir, 'model')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'body_shape_tree.pkl')

df = pd.read_csv(data_path)
X = df[['Shoulders', 'Bust', 'Waist', 'Hips', 'WHR']]
y = df['BodyShape']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = DecisionTreeClassifier(max_depth=4, random_state=0)
clf.fit(X_train, y_train)


joblib.dump(clf, model_path)
print(f"Model trained and saved to: {model_path}")
