import streamlit as st
import requests
from PIL import Image
import numpy as np
import joblib

# Load ML models (Replace with your actual models)
crop_model = joblib.load('models/crop_recommendation.pkl')
# yield_model = joblib.load('models/yield_prediction.pkl')
# fertilizer_model = joblib.load('models/fertilizer_recommendation.pkl')

def get_weather(city):
    api_key = "your_api_key_here"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url).json()
    return response

st.set_page_config(page_title="Bhoomi Dashboard", layout="wide")

st.title("🌱 Bhoomi - Integrated Crop Management System")

# Sidebar Navigation
menu = ["Crop Recommendation", "Identify Plant Disease", "Crop Yield Prediction", "Today's Weather", "Fertilizer Recommendation", "Smart Farming Guidance"]
choice = st.sidebar.selectbox("Select Feature", menu)

if choice == "Crop Recommendation":
    st.subheader("🌾 Crop Recommendation System")
    nitrogen = st.number_input("Nitrogen Level", min_value=0)
    phosphorus = st.number_input("Phosphorus Level", min_value=0)
    potassium = st.number_input("Potassium Level", min_value=0)
    ph = st.number_input("pH Level", min_value=0.0, max_value=14.0)
    rainfall = st.number_input("Rainfall (mm)")
    if st.button("Recommend Crop"):
        features = np.array([[nitrogen, phosphorus, potassium, ph, rainfall]])
        prediction = crop_model.predict(features)
        st.success(f"Recommended Crop: {prediction[0]}")

elif choice == "Identify Plant Disease":
    st.subheader("🦠 Plant Disease Identification")
    uploaded_file = st.file_uploader("Upload Plant Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.success("Processing Image... (Integrate ML Model Here)")

elif choice == "Crop Yield Prediction":
    st.subheader("📊 Crop Yield Prediction")
    area = st.number_input("Field Area (hectares)")
    rainfall = st.number_input("Rainfall (mm)")
    temperature = st.number_input("Temperature (°C)")
    if st.button("Predict Yield"):
        features = np.array([[area, rainfall, temperature]])
        prediction = yield_model.predict(features)
        st.success(f"Predicted Yield: {prediction[0]} tons")

elif choice == "Today's Weather":
    st.subheader("🌤️ Weather Forecast")
    city = st.text_input("Enter City Name")
    if st.button("Get Weather"):
        weather_data = get_weather(city)
        if weather_data.get('main'):
            st.write(f"Temperature: {weather_data['main']['temp']}°C")
            st.write(f"Weather: {weather_data['weather'][0]['description']}")
            st.write(f"Humidity: {weather_data['main']['humidity']}%")
        else:
            st.error("Invalid City Name!")

elif choice == "Fertilizer Recommendation":
    st.subheader("🧪 Fertilizer Recommendation")
    crop = st.text_input("Enter Crop Name")
    soil_type = st.text_input("Enter Soil Type")
    if st.button("Recommend Fertilizer"):
        features = np.array([[crop, soil_type]])
        prediction = fertilizer_model.predict(features)
        st.success(f"Recommended Fertilizer: {prediction[0]}")

elif choice == "Smart Farming Guidance":
    st.subheader("📚 Smart Farming Tips")
    st.write("✅ Use precision farming techniques.")
    st.write("✅ Implement soil testing for better yield.")
    st.write("✅ Adopt AI and IoT for automated irrigation.")
    st.write("✅ Practice crop rotation to maintain soil fertility.")
