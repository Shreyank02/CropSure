import streamlit as st
import pandas as pd
import numpy as np
import pickle

model_filename = "fertilizer.pkl"
try:
    with open(model_filename, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    model = None

# Streamlit App Title
st.title("Fertilizer Recommendation System")

crop_type_unique = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley',
       'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts']

# Define input fields for independent variables
st.write("### Enter the required values:")
temperature = st.number_input("Temperature (°C)", value=25.0)
humidity = st.number_input("Humidity (%)", value=50.0)
moisture = st.number_input("Moisture (%)", value=30.0)
soil_type = st.selectbox("Soil Type", ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey'])
crop_type = st.selectbox("Crop Type", sorted(crop_type_unique))
nitrogen = st.number_input("Nitrogen Level", value=10)
phosphorus = st.number_input("Phosphorus Level", value=10)
potassium = st.number_input("Potassium Level", value=10)

# Convert categorical inputs to numerical (assuming same encoding as training data)
soil_mapping = {"Sandy": 0, "Loamy": 1, "Clayey": 2, "Black":3, "Red":4}
crop_mapping = {"Wheat": 0, "Maize": 1, "Barley": 2, "Cotton": 3, "Ground Nuts": 4, "Sugarcane": 5, "Tobacco":6, "Paddy":7, "Millets":8, "Oil seeds":9, "Pulses":10, }
soil_type_encoded = soil_mapping[soil_type]
crop_type_encoded = crop_mapping[crop_type]

# Make Prediction
if st.button("Recommend Fertilizer"):
    if model:
        input_data = np.array([[temperature, humidity, moisture, soil_type_encoded, crop_type_encoded, nitrogen, phosphorus, potassium]])
        prediction = model.predict(input_data)
        st.write("### Recommended Fertilizer:", prediction[0])
    else:
        st.write("Model file not found. Please ensure the trained model is available.")
