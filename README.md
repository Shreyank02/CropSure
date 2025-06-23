# CropSure - AI-Powered Agricultural Assistance

CropSure is an AI-driven web application designed to assist farmers with essential agricultural insights. It provides recommendations for fertilizers, predicts crop yields, detects plant diseases, and offers other useful services to improve farming efficiency.

## 🚀 Features

- **Fertilizer Recommendation**: Get AI-based suggestions for optimal fertilizer use.
- **Crop Yield Prediction**: Predict crop yield based on historical data and input parameters.
- **Plant Disease Detection**: Identify plant diseases using image-based deep learning models.
- **Additional Services**: More AI-driven solutions to assist farmers.

## 🛠 Tech Stack

- **Frontend**: Streamlit (Python-based web framework)
- **Backend**: Machine Learning models (Scikit-learn, TensorFlow)
- **Models Used**:
  - RandomForestClassifier (Fertilizer Recommendation)
  - Pipeline Model (Crop Yield Prediction)
  - CNN (Plant Disease Detection)
- **Data Handling**: Pandas, NumPy
- **Deployment**: Streamlit-based web app

## 📦 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/CropSure.git
   cd CropSure
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure model files (`pipe.pkl`, `fertilizer.pkl`, `plant_disease_prediction_model.h5`) are present in the project directory.
5. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## 📝 Usage

1. Open the application in your browser.
2. Select a service from the navigation panel.
3. Input the required parameters.
4. Click on the prediction button to get insights.

## 🤖 Machine Learning Models

- **Fertilizer Recommendation Model**: Uses a trained RandomForest model.
- **Crop Yield Prediction Model**: Pipeline-based ML model.
- **Plant Disease Detection**: CNN model trained on agricultural datasets.

## 📂 Project Structure
```
CropSure/
│── app.py               # Main Streamlit App
│── plant-disease-prediction-cnn-1/
│   ├── plant_disease_prediction_model.h5  # Trained CNN model
│   ├── class_indices.json
│── crop yield prediction/
│   ├── pipe.pkl         # Crop yield prediction model
│── models/
│   ├── fertilizer.pkl   # Fertilizer recommendation model
│── requirements.txt     # Dependencies
│── README.md            # Project Documentation
```

## 🚀 Future Enhancements

- Improve model accuracy with more data.
- Deploy on cloud for broader accessibility.
- Add a chatbot for instant AI assistance.


---

Made with love ❤️ for farmers! 🌾

