import streamlit as st
import requests
import joblib
import numpy as np
from PIL import Image
from openai import OpenAI
from streamlit_lottie import st_lottie

# Load ML models
crop_model = joblib.load('models/crop_recommendation.pkl')
# yield_model = joblib.load('models/yield_prediction.pkl')
# fertilizer_model = joblib.load('models/fertilizer_recommendation.pkl')

# Function to fetch weather data
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

# OpenAI API for Smart Farming Guidance
client = OpenAI(api_key="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")

def get_smart_farming_info(crop, country):
    prompt = f"Provide detailed smart farming guidelines for {crop} in {country}, including fertilizers, time periods, and best practices."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Load Lottie Animation
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_farming = load_lottie_url("https://lottie.host/33f9d1b3-d10b-4d12-9a5e-7614f6a78fdc/animation.json")

# Streamlit Page Config
st.set_page_config(page_title="Bhoomi Dashboard", layout="wide")

# Custom CSS for Styling
st.markdown(
    """
    <style>
        .home-title { font-size: 36px; font-weight: bold; color: #4CAF50; text-align: center; padding-bottom: 10px; }
        .home-subtitle { font-size: 20px; text-align: center; color: #666; }
        .feature-card { background: #f8f9fa; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; text-align: center; }
        .sidebar .sidebar-content { transition: all 0.5s ease-in-out; }
        .sidebar:hover .sidebar-content { transform: scale(1.05); }
        .menu-button { display: block; width: 100%; padding: 10px; margin: 5px 0; font-size: 18px; font-weight: bold; text-align: left;
            background-color: #f0f0f0; border: none; border-radius: 5px; transition: background 0.3s ease; }
        .menu-button:hover { background-color: #d9d9d9; }
        .stApp {background: url("img/futuristic-technology-concept.jpg") no-repeat center center fixed; background-size: cover;}
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

# Home Page with Animations
if selected_menu == "Home":
    st.markdown("<h1 class='home-title'>🌱 Welcome to Bhoomi!</h1>", unsafe_allow_html=True)
    st.markdown("<p class='home-subtitle'>Your Smart AI-based Integrated Crop Management System</p>", unsafe_allow_html=True)

    image = Image.open("img/futuristic-technology-concept.jpg")  
    image = image.resize((400, 200))  
    st.image(image)
    
    st.write(
        "Bhoomi is designed to empower farmers with AI-driven insights and predictions for smarter agricultural decisions. "
        "From **crop recommendations** to **disease detection**, and **yield predictions**, Bhoomi integrates advanced machine learning models to enhance farming efficiency."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='feature-card'>🌾 <b>Crop Recommendation</b><br>Get AI-based crop suggestions</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='feature-card'>🦠 <b>Plant Disease Detection</b><br>Upload images & detect diseases</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='feature-card'>📊 <b>Crop Yield Prediction</b><br>Estimate crop yield based on weather</div>", unsafe_allow_html=True)

    if lottie_farming:
        st_lottie(lottie_farming, height=300, key="farming_anim")

# Crop Recommendation Page
elif selected_menu == "Crop Recommendation":
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
    if st.button("Recommend Fertilizer"):
        features = np.array([[crop, soil_type]])
        prediction = fertilizer_model.predict(features)
        st.success(f"Recommended Fertilizer: {prediction[0]}")

# Smart Farming Guidance
elif selected_menu == "Smart Farming Guidance":
    st.subheader("📚 Smart Farming Guidance")
    crop = st.text_input("Enter Crop Name")
    country = st.text_input("Enter Country Name")
    if st.button("Get Smart Farming Info"):
        guidance = get_smart_farming_info(crop, country)
        st.write(guidance)
        st.image(f"https://source.unsplash.com/600x400/?{crop}", caption=f"{crop}", use_container_width =True)

# Add other features here...

