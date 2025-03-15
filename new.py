import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import pickle
import pandas as pd

# Load Plant Disease Detection Model
model_path = "./plant-disease-prediction-cnn-1/plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open("./plant-disease-prediction-cnn-1/class_indices.json"))

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).resize(target_size)
    img_array = np.expand_dims(np.array(img).astype('float32') / 255., axis=0)
    return img_array

def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    return class_indices[str(np.argmax(predictions, axis=1)[0])]

# Load Fertilizer Recommendation Model
model_filename = "fertilizer.pkl"
try:
    with open(model_filename, "rb") as file:
        fertilizer_model = pickle.load(file)
except FileNotFoundError:
    fertilizer_model = None

# Load Crop Yield Prediction Model
yield_model_filename = "./crop yield prediction/pipe.pkl"
try:
    with open(yield_model_filename, "rb") as file:
        yield_model = pickle.load(file)
except FileNotFoundError:
    yield_model = None

# Define categorical options
crop_options = ['Arecanut', 'Arhar/Tur', 'Castor seed', 'Coconut ', 'Cotton(lint)',
       'Dry chillies', 'Gram', 'Jute', 'Linseed', 'Maize', 'Mesta',
       'Niger seed', 'Onion', 'Other Rabi pulses', 'Potato',
       'Rapeseed & Mustard', 'Rice', 'Sesamum', 'Small millets',
       'Sugarcane', 'Sweet potato', 'Tapioca', 'Tobacco', 'Turmeric',
       'Wheat', 'Bajra', 'Black pepper', 'Cardamom', 'Coriander',
       'Garlic', 'Ginger', 'Groundnut', 'Horse-gram', 'Jowar', 'Ragi',
       'Cashewnut', 'Banana', 'Soyabean', 'Barley', 'Khesari', 'Masoor',
       'Moong(Green Gram)', 'Other Kharif pulses', 'Sannhamp',
       'Sunflower', 'Urad', 'Peas & beans (Pulses)', 'Safflower',
       'Other oilseeds', 'Other Cereals', 'Cowpea(Lobia)',
       'Oilseeds total', 'Guar seed', 'Other Summer Pulses', 'Moth']

season_options = ['Whole Year', 'Kharif', 'Rabi', 'Autumn', 'Summer', 'Winter']

state_options = ['Assam', 'Karnataka', 'Kerala', 'Meghalaya', 'West Bengal',
       'Puducherry', 'Goa', 'Andhra Pradesh', 'Tamil Nadu', 'Odisha',
       'Bihar', 'Gujarat', 'Madhya Pradesh', 'Maharashtra', 'Mizoram',
       'Punjab', 'Uttar Pradesh', 'Haryana', 'Himachal Pradesh',
       'Tripura', 'Nagaland', 'Chhattisgarh', 'Uttarakhand', 'Jharkhand',
       'Delhi', 'Manipur', 'Jammu and Kashmir', 'Telangana',
       'Arunachal Pradesh', 'Sikkim']

crop_type_unique = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley',
       'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts']

# Define categorical mappings
crop_mapping = {crop: idx for idx, crop in enumerate(crop_options)}
season_mapping = {season: idx for idx, season in enumerate(season_options)}
state_mapping = {state: idx for idx, state in enumerate(state_options)}
year_mapping = {year: idx for idx, year in enumerate(['90s', '2000s', '2010s'])}

def fertilizer_app():
    st.title("Fertilizer Recommendation System")
    st.write("### Enter the required values:")
    temperature = st.number_input("Temperature (°C)", value=25.0)
    humidity = st.number_input("Humidity (%)", value=50.0)
    moisture = st.number_input("Moisture (%)", value=30.0)
    soil_type = st.selectbox("Soil Type", ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey'])
    crop_type = st.selectbox("Crop Type", sorted(crop_type_unique))
    nitrogen = st.number_input("Nitrogen Level", value=10)
    phosphorus = st.number_input("Phosphorus Level", value=10)
    potassium = st.number_input("Potassium Level", value=10)

    soil_mapping = {"Sandy": 0, "Loamy": 1, "Clayey": 2, "Black":3, "Red":4}
    crop_mapping = {"Wheat": 0, "Maize": 1, "Barley": 2, "Cotton": 3, "Ground Nuts": 4, "Sugarcane": 5, "Tobacco":6, "Paddy":7, "Millets":8, "Oil seeds":9, "Pulses":10, }
    soil_type_encoded = soil_mapping[soil_type]
    crop_type_encoded = crop_mapping[crop_type]

    if st.button("Recommend Fertilizer"):
        if fertilizer_model:
            input_data = np.array([[temperature, humidity, moisture, soil_type_encoded, crop_type_encoded, nitrogen, phosphorus, potassium]])
            prediction = fertilizer_model.predict(input_data)
            st.write("### Recommended Fertilizer:", prediction[0])
        else:
            st.write("Model file not found. Please ensure the trained model is available.")

def disease_app():
    st.title('Plant Disease Classifier')
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image.resize((150, 150)))
        with col2:
            if st.button('Classify'):
                prediction = predict_image_class(model, uploaded_image, class_indices)
                st.success(f'Prediction: {str(prediction)}')

def yield_prediction_app():
    st.title("Crop Yield Prediction")
    st.write("### Enter the required values:")
    crop = st.selectbox("Select Crop", list(crop_mapping.keys()))
    season = st.selectbox("Select Season", list(season_mapping.keys()))
    state = st.selectbox("Select State", list(state_mapping.keys()))
    area = st.number_input("Area of Land (in hectares)", value=1.0)
    production = st.number_input("Total Production (in tons)", value=1.0)
    rainfall = st.number_input("Annual Rainfall (in mm)", value=1000.0)
    fertilizer = st.number_input("Fertilizer Usage (kg per hectare)", value=500.0)
    pesticide = st.number_input("Pesticide Usage (kg per hectare)", value=10.0)
    year_interval = st.selectbox("Select Year Interval", list(year_mapping.keys()))
    
    Input_Per_Unit_Area = (fertilizer + pesticide) / area
    
    if st.button("Predict Crop Yield"):
        if yield_model:
                
                input_data = pd.DataFrame({
                    'Crop': [crop_mapping[crop]],
                    'Season': [season_mapping[season]],
                    'State': [state_mapping[state]],
                    'Area': [float(area)],
                    'Production': [float(production)],
                    'Annual_Rainfall': [float(rainfall)],
                    'Input_Per_Unit_Area': [float(Input_Per_Unit_Area)],
                    'Year_Interval': [year_mapping[year_interval]]
                })
                
                # Debugging: Check for NaN or incorrect types
                st.write("### Debugging Input Data:")
                st.write(input_data)

                # Ensure no NaN values
                if input_data.isnull().values.any():
                    st.error("Error: Some input values are NaN. Please check your inputs.")
                else:

                    try:
                        prediction = yield_model.predict(input_data)
                        st.write("### Predicted Crop Yield (in tons):", round(prediction[0], 2))
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
        else:
            st.write("Model file not found. Please ensure the trained model is available.")


# Main App with Navigation
st.set_page_config(page_title="CropSure - Smart Farming Solutions", layout="wide")
st.sidebar.title("**CropSure Services**")
option = st.sidebar.radio("Choose a service:", ["Fertilizer Recommendation", "Disease Detection", "Crop Yield Prediction"])

if option == "Fertilizer Recommendation":
    fertilizer_app()
elif option == "Disease Detection":
    disease_app()
elif option == "Crop Yield Prediction":
    yield_prediction_app()

st.sidebar.markdown("---")
st.sidebar.markdown("**CropSure: Empowering Farmers with AI**")
st.sidebar.markdown("**A smart farming solution that helps farmers and agricultural sector grow. We here at CropSure solve new-age farming problems with new-age solutions.**")
st.sidebar.markdown("**Integrating AI to the roots of farming**")
