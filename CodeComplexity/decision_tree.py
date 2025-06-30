"""
Decision Tree Process:
1. Data Preparation: 
   - Reads local CSV file with user features (age, income, etc.) and target (product category)
   - Separates features (X) from target labels (y)

2. Preprocessing:
   - Automatically one-hot encodes categorical features (gender, location, etc.)
   - Keeps numerical features unchanged

3. Model Training: 
   - Builds a decision tree that learns optimal split points
   - Uses information gain to determine most important features

4. Visualization:
   - Outputs flowchart showing decision rules with simplified labels
   - Each node displays splitting condition and class distribution
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import os


file_path = "/path/to/syntheticData.csv"            #change this to your local path
df = pd.read_csv(file_path)

X = df.drop(columns=['user_id', 'product'])         ## Drop user_id and product columns from features
y = df['product']

# Preprocessing Setup , change all categorical features to one-hot encoding
categorical_cols = ['gender', 'location', 'education', 'job', 'loyalty_status']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Create and Train Pipeline
dt_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(
        max_depth=3,
        min_samples_split=5,
        random_state=24  
    ))
])

dt_model.fit(X, y)
print("=== Model Training Complete ===")

# Simplify feature names for visualization, because each feature name is a combination of
# the original column name and the one-hot encoded category
feature_names = dt_model.named_steps['preprocessor'].get_feature_names_out()
simplified_names = [name.split('__')[-1].replace('_', '=') if 'cat__' in name else name 
                   for name in feature_names]


#Visualize Decision Tree
plt.figure(figsize=(24, 12))
plot_tree(
    dt_model.named_steps['classifier'],
    feature_names=simplified_names,
    class_names=dt_model.classes_,
    filled=True,
    rounded=True,
    fontsize=10,
    proportion=True
)
plt.title("Decision Tree Visualization - 3 Levels Deep", fontsize=14, pad=20)
plt.show()