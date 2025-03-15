import streamlit as st
import requests
from PIL import Image
import numpy as np
import joblib
from openai import OpenAI
import io
from tensorflow.keras.preprocessing import image as keras_image
import plotly.express as px
import pandas as pd

# Must be the first Streamlit command
st.set_page_config(page_title="Bhoomi Dashboard", layout="wide", initial_sidebar_state="expanded")

# Initialize OpenAI client with the correct API key
client = OpenAI(api_key="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")  # Replace with your actual OpenAI API key

# Load ML models with caching and error handling
@st.cache_resource
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        return None

crop_model = load_model('models/crop_recommendation.pkl')
yield_model = None  # Placeholder
fertilizer_model = None  # Placeholder

# Placeholder for disease model
@st.cache_resource
def load_disease_model():
    return None

disease_model = load_disease_model()

# Weather API function
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

@st.cache_data
def get_smart_farming_info(crop, country):
    try:
        prompt = f"Provide detailed smart farming guidelines for {crop} in {country}, including fertilizers, time periods, and best practices."
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error fetching smart farming guidance: {str(e)}"

def predict_disease(image):
    if disease_model is None:
        return "Disease detection model not loaded (placeholder)"
    img = keras_image.img_to_array(image.resize((224, 224))) / 255.0
    img = np.expand_dims(img, axis=0)
    return "Disease Name (placeholder)"

# Custom CSS for modern UI
st.markdown(
    """
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f0f2f5;
        }
        .main {
            padding: 20px;
        }
        .sidebar .sidebar-content {
            background-color: #4a90e2;
            color: white;
            padding: 20px;
            border-radius: 10px;
        }
        .sidebar .sidebar-content .menu-button {
            background-color: #357abd;
            color: white;
            border: none;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            width: 100%;
            text-align: left;
        }
        .sidebar .sidebar-content .menu-button:hover {
            background-color: #2e6da4;
        }
        .stButton>button {
            background-color: #4a90e2;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #357abd;
        }
        .card {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            margin: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .card h3 {
            color: #4a90e2;
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# User registration session
if 'user_info' not in st.session_state:
    st.title("🌱 Bhoomi - Farmer Registration")
    with st.form("user_form"):
        name = st.text_input("Full Name")
        mobile = st.text_input("Mobile Number", help="e.g., 9876543210")
        place = st.text_input("Place")
        submitted = st.form_submit_button("Submit")
    if submitted:
        if name and mobile and place:
            st.session_state.user_info = {"name": name, "mobile": mobile, "place": place}
            st.success("Registration successful! Redirecting to dashboard...")
            st.session_state.menu = "Home"  # Jump to main app
            st.rerun()
        else:
            st.error("Please fill in all fields.")
else:
    st.title(f"🌱 Bhoomi - Welcome {st.session_state.user_info['name']}")
    st.markdown("Your personalized farming dashboard.")

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

    # Dummy data for charts
    df_yield = pd.DataFrame({
        "Date": ["2025-03-01", "2025-03-02", "2025-03-03", "2025-03-04", "2025-03-05"],
        "Yield": [2, 3, 1, 4, 2]
    })
    df_income = pd.DataFrame({"Value": [0]})
    df_expenses = pd.DataFrame({"Value": [0]})
    df_yield_total = pd.DataFrame({"Value": [0]})
    df_profit = pd.DataFrame({"Value": [0]})

    # Home page with statistics
    if selected_menu == "Home":
        st.subheader("Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="card"><h3>Yield over time</h3></div>', unsafe_allow_html=True)
            fig = px.line(df_yield, x="Date", y="Yield", title="")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown('<div class="card"><h3>₹</h3><p>Total Income over time</p><p>0</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="card"><h3>🛒</h3><p>Total Expenses over time</p><p>0</p></div>', unsafe_allow_html=True)
        col4, col5 = st.columns(2)
        with col4:
            st.markdown('<div class="card"><h3>🌾</h3><p>Total Yield over time</p><p>0</p></div>', unsafe_allow_html=True)
        with col5:
            st.markdown('<div class="card"><h3>💰</h3><p>Total Profit over time</p><p>0</p></div>', unsafe_allow_html=True)
        st.subheader("Crop Yield Distribution")
        st.write("Placeholder for distribution chart (to be implemented with real data)")

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
                    features = np.array([[hash(crop) % 100, hash(soil_type) % 100]])
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

