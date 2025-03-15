import streamlit as st
import requests
from PIL import Image
import numpy as np
import joblib
from openai import OpenAI
import io
from tensorflow.keras.preprocessing import image as keras_image  # Placeholder for disease model

# Must be the first Streamlit command
st.set_page_config(page_title="Bhoomi Dashboard", layout="wide", initial_sidebar_state="expanded")

# Initialize OpenAI client (replace with your actual key)
client = OpenAI(api_key="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")

# Load ML models with caching and error handling
@st.cache_resource
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        return None

crop_model = load_model('models/crop_recommendation.pkl')
# Commenting out models that aren't ready yet
# yield_model = load_model('models/yield_prediction.pkl')
# fertilizer_model = load_model('models/fertilizer_recommendation.pkl')
yield_model = None  # Placeholder
fertilizer_model = None  # Placeholder

# Placeholder for disease model (replace with actual model loading if available)
@st.cache_resource
def load_disease_model():
    # Example: return load_model('models/disease_classifier.h5')
    return None  # Placeholder

disease_model = load_disease_model()

# Weather API function with improved error handling
def get_weather(zip_code, country_code="IN"):
    api_key = "f938f65079af3e9bd2414c6556df724b"
    url = f"http://api.openweathermap.org/geo/1.0/zip?zip={zip_code},{country_code}&appid={api_key}"
    try:
        response = requests.get(url).json()
        if 'lat' not in response or 'lon' not in response:
            return {"error": "Invalid ZIP code or country code"}
        lat, lon = response['lat'], response['lon']
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        weather_response = requests.get(weather_url).json()
        return weather_response
    except requests.RequestException:
        return {"error": "Failed to connect to weather service"}

# Smart farming guidance with caching
@st.cache_data
def get_smart_farming_info(crop, country):
    prompt = f"Provide detailed smart farming guidelines for {crop} in {country}, including fertilizers, time periods, and best practices."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Placeholder for disease prediction
def predict_disease(image):
    if disease_model is None:
        return "Disease detection model not loaded (placeholder)"
    img = keras_image.img_to_array(image.resize((224, 224))) / 255.0
    img = np.expand_dims(img, axis=0)
    # prediction = disease_model.predict(img)  # Replace with actual prediction
    return "Disease Name (placeholder)"  # Replace with actual output

# Custom CSS for a modern, good-looking UI
st.markdown(
    """
    <style>
        body {
            font-family: 'Arial', sans-serif;
        }
        .main {
            background-color: #f4f7f6;
            padding: 20px;
            border-radius: 10px;
        }
        .sidebar .sidebar-content {
            background-color: #2e7d32;
            color: white;
            padding: 20px;
            border-radius: 10px;
            transition: all 0.3s ease-in-out;
        }
        .sidebar .sidebar-content:hover {
            transform: scale(1.02);
        }
        .menu-button {
            display: block;
            width: 100%;
            padding: 12px;
            margin: 8px 0;
            font-size: 16px;
            font-weight: bold;
            text-align: left;
            background-color: #4caf50;
            color: white;
            border: none;
            border-radius: 8px;
            transition: background 0.3s ease;
        }
        .menu-button:hover {
            background-color: #388e3c;
        }
        .stButton>button {
            background-color: #4caf50;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #388e3c;
        }
        .stNumberInput, .stTextInput {
            border-radius: 8px;
            padding: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# App title and intro
st.title("🌱 Bhoomi - Integrated Crop Management System")
st.markdown("A smart farming tool to optimize your agricultural practices.")

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
    if st.sidebar.button(label, key=key, help=f"Go to {key}"):
        st.session_state['menu'] = key

selected_menu = st.session_state['menu']

# Main content based on selected menu
if selected_menu == "Home":
    st.subheader("Welcome to Bhoomi!")
    st.write("Your one-stop solution for modern farming. Use the sidebar to navigate through our features.")
    st.image("https://source.unsplash.com/800x400/?farming", use_column_width=True)

elif selected_menu == "Crop Recommendation":
    st.subheader("🌾 Crop Recommendation System")
    with st.form("crop_form"):
        nitrogen = st.number_input("Nitrogen Level (N)", min_value=0, value=0)
        phosphorus = st.number_input("Phosphorus Level (P)", min_value=0, value=0)
        potassium = st.number_input("Potassium Level (K)", min_value=0, value=0)
        ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.0)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=0.0)
        submitted = st.form_submit_button("Recommend Crop")
    if submitted and crop_model:
        if all([nitrogen, phosphorus, potassium, rainfall]):
            features = np.array([[nitrogen, phosphorus, potassium, ph, rainfall]])
            with st.spinner("Analyzing soil data..."):
                prediction = crop_model.predict(features)
            st.success(f"Recommended Crop: **{prediction[0]}**")
        else:
            st.error("Please fill in all fields.")
    elif submitted and not crop_model:
        st.warning("Crop recommendation model not available yet.")

elif selected_menu == "Identify Plant Disease":
    st.subheader("🦠 Plant Disease Identification")
    uploaded_file = st.file_uploader("Upload Plant Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Analyzing image..."):
            disease = predict_disease(image)
        st.success(f"Detected Disease: **{disease}**")

elif selected_menu == "Crop Yield Prediction":
    st.subheader("📊 Crop Yield Prediction")
    with st.form("yield_form"):
        area = st.number_input("Field Area (hectares)", min_value=0.0, value=0.0)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=0.0)
        temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, value=25.0)
        submitted = st.form_submit_button("Predict Yield")
    if submitted:
        if yield_model:
            if all([area, rainfall, temperature]):
                features = np.array([[area, rainfall, temperature]])
                with st.spinner("Predicting yield..."):
                    prediction = yield_model.predict(features)
                st.success(f"Predicted Yield: **{prediction[0]:.2f} tons**")
            else:
                st.error("Please fill in all fields.")
        else:
            st.warning("Yield prediction model not available yet. Placeholder output: **5.0 tons**")

elif selected_menu == "Today's Weather":
    st.subheader("🌤️ Weather Forecast")
    with st.form("weather_form"):
        zip_code = st.text_input("Enter ZIP Code", help="e.g., 110001 for Delhi")
        country_code = st.text_input("Enter Country Code", value="IN", help="e.g., IN for India")
        submitted = st.form_submit_button("Get Weather")
    if submitted:
        with st.spinner("Fetching weather data..."):
            weather_data = get_weather(zip_code, country_code)
        if "error" in weather_data:
            st.error(weather_data["error"])
        elif weather_data.get('main'):
            city_name = weather_data.get('name', 'Unknown Location')
            st.write(f"**Location**: {city_name}")
            st.write(f"**Temperature**: {weather_data['main']['temp']}°C")
            st.write(f"**Weather**: {weather_data['weather'][0]['description'].capitalize()}")
            st.write(f"**Humidity**: {weather_data['main']['humidity']}%")
        else:
            st.error("Could not retrieve weather data.")

elif selected_menu == "Fertilizer Recommendation":
    st.subheader("🧪 Fertilizer Recommendation")
    with st.form("fertilizer_form"):
        crop = st.text_input("Enter Crop Name", help="e.g., Rice")
        soil_type = st.text_input("Enter Soil Type", help="e.g., Sandy")
        submitted = st.form_submit_button("Recommend Fertilizer")
    if submitted:
        if fertilizer_model:
            if crop and soil_type:
                # Placeholder: assumes model accepts encoded inputs; adjust as needed
                features = np.array([[hash(crop) % 100, hash(soil_type) % 100]])  # Dummy encoding
                with st.spinner("Analyzing..."):
                    prediction = fertilizer_model.predict(features)
                st.success(f"Recommended Fertilizer: **{prediction[0]}**")
            else:
                st.error("Please fill in all fields.")
        else:
            st.warning("Fertilizer recommendation model not available yet. Placeholder output: **NPK 20-20-20**")

elif selected_menu == "Smart Farming Guidance":
    st.subheader("📚 Smart Farming Guidance")
    with st.form("guidance_form"):
        crop = st.text_input("Enter Crop Name", help="e.g., Wheat")
        country = st.text_input("Enter Country Name", help="e.g., India")
        submitted = st.form_submit_button("Get Smart Farming Info")
    if submitted:
        if crop and country:
            with st.spinner("Fetching guidance..."):
                guidance = get_smart_farming_info(crop, country)
            st.markdown(guidance)
            st.image(f"https://source.unsplash.com/600x400/?{crop}", caption=f"{crop.capitalize()}", use_column_width=True)
        else:
            st.error("Please fill in all fields.")

# Footer
st.markdown("---")
st.markdown("Developed by Aditya Pise | © 2025")
