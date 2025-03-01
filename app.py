import streamlit as st
import requests
from PIL import Image
import numpy as np
import joblib

# Load ML models (Replace with your actual models)
crop_model = joblib.load('models/crop_recommendation.pkl')
# yield_model = joblib.load('models/yield_prediction.pkl')
# fertilizer_model = joblib.load('models/fertilizer_recommendation.pkl')

def get_weather(zip_code, country_code="IN"):
    api_key = "your_api_key_here"
    url = f"http://api.openweathermap.org/geo/1.0/zip?zip={zip_code},{country_code}&appid={api_key}"
    response = requests.get(url).json()
    if 'lat' in response and 'lon' in response:
        lat, lon = response['lat'], response['lon']
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        weather_response = requests.get(weather_url).json()
        return weather_response
    return None

st.set_page_config(page_title="Bhoomi Dashboard", layout="wide")

st.markdown(
    """
    <style>
        .sidebar .sidebar-content {
            transition: all 0.5s ease-in-out;
        }
        .sidebar:hover .sidebar-content {
            transform: scale(1.05);
        }
        .menu-button {
            display: block;
            width: 100%;
            padding: 10px;
            margin: 5px 0;
            font-size: 18px;
            font-weight: bold;
            text-align: left;
            background-color: #f0f0f0;
            border: none;
            border-radius: 5px;
            transition: background 0.3s ease;
        }
        .menu-button:hover {
            background-color: #d9d9d9;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌱 Bhoomi - Integrated Crop Management System")

# Sidebar Navigation
st.sidebar.title("🌍 Navigation")

if 'menu' not in st.session_state:
    st.session_state['menu'] = "Home"

menu_options = {
    "Home": "🏠 Home",
    "Crop Recommendation": "🌾 Crop Recommendation",
    "Identify Plant Disease": "🦠 Identify Plant Disease",
    "Crop Yield Prediction": "📊 Crop Yield Prediction",
    "Today's Weather": "🌤️ Today's Weather",
    "Fertilizer Recommendation": "🧪 Fertilizer Recommendation",
    "Smart Farming Guidance": "📚 Smart Farming Guidance"
}

for key, label in menu_options.items():
    if st.sidebar.button(label, key=key):
        st.session_state['menu'] = key

selected_menu = st.session_state['menu']

if selected_menu == "Crop Recommendation":
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

elif selected_menu == "Identify Plant Disease":
    st.subheader("🦠 Plant Disease Identification")
    uploaded_file = st.file_uploader("Upload Plant Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.success("Processing Image... (Integrate ML Model Here)")

elif selected_menu == "Crop Yield Prediction":
    st.subheader("📊 Crop Yield Prediction")
    area = st.number_input("Field Area (hectares)")
    rainfall = st.number_input("Rainfall (mm)")
    temperature = st.number_input("Temperature (°C)")
    if st.button("Predict Yield"):
        features = np.array([[area, rainfall, temperature]])
        prediction = yield_model.predict(features)
        st.success(f"Predicted Yield: {prediction[0]} tons")

elif selected_menu == "Today's Weather":
    st.subheader("🌤️ Weather Forecast")
    zip_code = st.text_input("Enter ZIP Code")
    country_code = st.text_input("Enter Country Code (e.g., IN for India)", value="IN")
    if st.button("Get Weather"):
        weather_data = get_weather(zip_code, country_code)
        if weather_data and weather_data.get('main'):
            st.write(f"Temperature: {weather_data['main']['temp']}°C")
            st.write(f"Weather: {weather_data['weather'][0]['description']}")
            st.write(f"Humidity: {weather_data['main']['humidity']}%")
        else:
            st.error("Invalid ZIP Code or Country Code!")

elif selected_menu == "Fertilizer Recommendation":
    st.subheader("🧪 Fertilizer Recommendation")
    crop = st.text_input("Enter Crop Name")
    soil_type = st.text_input("Enter Soil Type")
    if st.button("Recommend Fertilizer"):
        features = np.array([[crop, soil_type]])
        prediction = fertilizer_model.predict(features)
        st.success(f"Recommended Fertilizer: {prediction[0]}")

elif selected_menu == "Smart Farming Guidance":
    st.subheader("📚 Smart Farming Tips")
    st.write("✅ Use precision farming techniques.")
    st.write("✅ Implement soil testing for better yield.")
    st.write("✅ Adopt AI and IoT for automated irrigation.")
    st.write("✅ Practice crop rotation to maintain soil fertility.")
