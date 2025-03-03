import streamlit as st
import requests
from PIL import Image
import numpy as np
import joblib

# Load ML models (Ensure these exist in your project)
crop_model = joblib.load('models/crop_recommendation.pkl')

# Function to get weather data
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

# Page Config
st.set_page_config(page_title="Bhoomi Dashboard", layout="wide")

# CSS for Centering Content
st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #2E8B57;
        }
        .center-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .stButton > button {
            width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Centering Content
with st.container():
    st.markdown('<h1 class="main-title">🌱 Bhoomi - Integrated Crop Management System</h1>', unsafe_allow_html=True)
    
    # Display Image
    farm_image = Image.open("farm.jpg")
    st.image(farm_image, caption="Sustainable Farming", use_container_width=True)

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

selected_menu = st.sidebar.radio("Select an Option", list(menu_options.keys()))

# Center Content
with st.container():
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
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.success("Processing Image... (Integrate ML Model Here)")

    elif selected_menu == "Crop Yield Prediction":
        st.subheader("📊 Crop Yield Prediction")
        area = st.number_input("Field Area (hectares)")
        rainfall = st.number_input("Rainfall (mm)")
        temperature = st.number_input("Temperature (°C)")
        if st.button("Predict Yield"):
            st.success("Yield Prediction Model Not Yet Integrated!")

    elif selected_menu == "Today's Weather":
        st.subheader("🌤️ Weather Forecast")
        zip_code = st.text_input("Enter ZIP Code")
        country_code = st.text_input("Enter Country Code (e.g., IN for India)", value="IN")
        if st.button("Get Weather"):
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
        st.success("Fertilizer Recommendation Model Not Yet Integrated!")

    elif selected_menu == "Smart Farming Guidance":
        st.subheader("📚 Smart Farming Guidance")
        st.success("Smart Farming Guidance Feature Not Yet Integrated!")
