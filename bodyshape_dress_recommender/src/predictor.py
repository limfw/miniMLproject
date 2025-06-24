import os
import pandas as pd
import joblib
from src.style_mapping import get_style_recommendation

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
model_path = os.path.join(parent_dir, 'model', 'body_shape_tree.pkl')

model = joblib.load(model_path)

def predict_body_shape(input_data):
    df = pd.DataFrame([input_data])
    df['WHR'] = df['Waist'] / df['Hips']

    X = df[['Shoulders', 'Bust', 'Waist', 'Hips', 'WHR']]
    prediction = model.predict(X)[0]
    recommendation = get_style_recommendation(prediction)
    return prediction, recommendation
