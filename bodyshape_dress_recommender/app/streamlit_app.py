import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

from src.predictor import predict_body_shape
import streamlit as st


st.set_page_config(page_title="Dress Recommendation by Body Shape", layout="centered")
st.title("Body Shape & Dress Style Recommender")
st.markdown("Enter your body measurements to get a personalized recommendation.")

shoulders = st.number_input("Shoulder Width (inches)", min_value=30.0, max_value=60.0, value=39.0)
bust = st.number_input("Bust (inches)", min_value=30.0, max_value=60.0, value=36.0)
waist = st.number_input("Waist (inches)", min_value=20.0, max_value=50.0, value=29.0)
hips = st.number_input("Hips (inches)", min_value=30.0, max_value=60.0, value=40.0)

if st.button("Predict My Body Shape"):
    input_data = {
        "Shoulders": shoulders,
        "Bust": bust,
        "Waist": waist,
        "Hips": hips
    }

    shape, recommendation = predict_body_shape(input_data)

    st.success(f"Predicted Body Shape: **{shape}**")
    st.markdown(f"**Style Recommendation:**\n\n{recommendation}")
