from src.predictor import predict_body_shape

input_data = {
    "Shoulders": 39,
    "Bust": 36,
    "Waist": 29,
    "Hips": 40
}

shape, style = predict_body_shape(input_data)
print("Predicted Body Shape:", shape)
print("Recommended Dress Style:", style)
