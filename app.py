import streamlit as st
import pandas as pd
import joblib

# Load the trained model and label encoder
model = joblib.load('disease_prediction_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Replace these with the 10 features used in training
feature_names = [
    'Glucose', 'Cholesterol', 'Hemoglobin', 'Platelets', 
    'White Blood Cells', 'Red Blood Cells', 
    'Systolic Blood Pressure', 'Diastolic Blood Pressure', 
    'BMI', 'HbA1c'
]

# Title of the app
st.title("Disease Prediction App")
st.write("Enter the features to predict the disease:")

# Create input fields for each feature
input_features = {}
for feature in feature_names:
    input_features[feature] = st.number_input(f"Enter {feature}", value=0.0)

# Convert input to DataFrame
input_data = pd.DataFrame(input_features, index=[0])

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    disease = label_encoder.inverse_transform(prediction)

    # Display the prediction
    st.write(f"Predicted Disease: {disease[0]}")
