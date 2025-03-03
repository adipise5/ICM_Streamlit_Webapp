import streamlit as st
import requests
from PIL import Image
import numpy as np
import joblib
from openai import OpenAI

# Load ML models (Replace with your actual models)
crop_model = joblib.load('models/crop_recommendation.pkl')
# yield_model = joblib.load('models/yield_prediction.pkl')
# fertilizer_model = joblib.load('models/fertilizer_recommendation.pkl')

def get_weather(zip_code, country_code="IN"):
    api_key = "f938f65079af3e9bd2414c6556df724b"
    url = f"http://api.openweathermap.org/geo/1.0/zip?zip={zip_code},{country_code}&appid={api_key}"
    response = requests.get(url).json()
    if 'lat' in response and 'lon' in response:
        lat, lon = response['lat'], response['lon']
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        weather_response = requests.get(weather_url).json()
        return weather_response
    return None
    
client = OpenAI(api_key="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")

def get_smart_farming_info(crop, country):
    prompt = f"Provide detailed smart farming guidelines for {crop} in {country}, including fertilizers, time periods, and best practices."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

st.set_page_config(page_title="Bhoomi Dashboard", layout="wide")

st.markdown(
    """
    <style>
        body {
            background-color: #e9f5e9;
            font-family: 'Arial', sans-serif;
        }
        .sidebar {
            background: linear-gradient(180deg, #4CAF50 0%, #2E7D32 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
        }
        .sidebar .sidebar-content {
            transition: all 0.5s ease-in-out;
        }
        .sidebar:hover .sidebar-content {
            transform: scale(1.05);
        }
        .menu-button {
            display: block;
            width: 100%;
            padding: 12px;
            margin: 8px 0;
            font-size: 18px;
            font-weight: bold;
            text-align: left;
            background-color: transparent;
            color: white;
            border: none;
            border-radius: 5px;
            transition: background 0.3s ease, transform 0.2s ease;
        }
        .menu-button:hover {
            background-color: rgba(255, 255, 255, 0.2);
            transform: translateY(-2px);
        }
        h1 {
            color: #4CAF50;
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
    if st.sidebar.button(label, key=key, help=f"Go to {label} section", css_class="menu-button"):
        st.session_state['menu'] = key

selected_menu = st.session_state['menu']

if selected_menu == "Crop Recommendation":
    st.subheader("🌾 Crop Recommendation System")
    nitrogen = st.number_input("Nitrogen Level", min_value=0)
    phosphorus = st.number_input("Phosphorus Level", min_value=0)
    potassium = st.number_input("Potassium Level", min_value=0)
    ph = st.number_input("pH Level", min_value=0.0, max_value=14.0)
    rainfall = st.number_input("Rainfall (mm)")
    if st.button("Recommend Crop", css_class="menu-button"):
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
    if st.button("Predict Yield", css_class="menu-button"):
        features = np.array([[area, rainfall, temperature]])
        prediction = yield_model.predict(features)
        st.success(f"Predicted Yield: {prediction[0]} tons")

elif selected_menu == "Today's Weather":
    st.subheader("🌤️ Weather Forecast")
    zip_code = st.text_input("Enter ZIP Code")
    country_code = st.text_input("Enter Country Code (e.g., IN for India)", value="IN")
    if st.button("Get Weather", css_class="menu-button"):
        weather_data = get_weather(zip_code, country_code)
        if weather_data and weather_data.get('main'):
            city_name = weather_data.get('name', 'Unknown Location')
            st.write(f"Location: {city_name}")
            st.write(f"Temperature: {weather_data['main']['temp']}°C")
            st.write(f"Weather: {weather_data['weather'][0]['description']}")
            st.write(f"Humidity: {weather_data['main']['humidity']}%")
        else:
            st.error("Invalid ZIP Code or Country Code!")

elif selected_menu == "Fertilizer Recommendation":
    st.subheader("🧪 Fertilizer Recommendation")
    crop = st.text_input("Enter Crop Name")
    soil_type = st.text_input("Enter Soil Type")
    if st.button("Recommend Fertilizer", css_class="menu-button"):
        features = np.array([[crop, soil_type]])
        prediction = fertilizer_model.predict(features)
        st.success(f"Recommended Fertilizer: {prediction[0]}")

elif selected_menu == "Smart Farming Guidance":
    st.subheader("📚 Smart Farming Guidance")
    crop = st.text_input("Enter Crop Name")
    country = st.text_input("Enter Country Name")
    if st.button("Get Smart Farming Info", css_class="menu-button"):
        guidance = get_smart_farming_info(crop, country)
        st.write(guidance)
        st.image(f"https://source.unsplash.com/600x400/?{crop}", caption=f"{crop}", use_column_width=True)

# Footer for contact information and credits
st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #4CAF50;
            color: white;
            text-align: center;
            padding: 10px 0;
        }
    </style>
    <div class="footer">
        <p>Contact us: support@bhoomidashboard.com | © 2023 Bhoomi Dashboard</p>
    </div>
    """,
    unsafe_allow_html=True
)
