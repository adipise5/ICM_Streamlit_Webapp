import streamlit as st
import requests
from PIL import Image
import numpy as np
import joblib
import os
import io
from tensorflow.keras.preprocessing import image as keras_image
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Must be the first Streamlit command
st.set_page_config(page_title="Bhoomi Dashboard", layout="wide", initial_sidebar_state="expanded")

# xAI API Key from environment variable
XAI_API_KEY = os.getenv("XAI_API_KEY", "xai-Es0CPO6ARiKBHkmHpO4PdiMGFJUjDDqFq6mNWJeQVLdeF8bv9SpezkYC0nQCC9R3tZChgopAQME9bpmo")
XAI_API_URL = "https://api.x.ai/v1/completions"  # Hypothetical URL; replace with actual endpoint

# Load ML models with caching and error handling
@st.cache_resource
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"🚨 Model file not found: {model_path}")
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
            return {"error": "🚫 Invalid ZIP code or country code"}
        lat, lon = response['lat'], response['lon']
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        weather_response = requests.get(weather_url).json()
        return weather_response
    except requests.RequestException:
        return {"error": "🌐 Failed to connect to weather service"}

@st.cache_data
def get_smart_farming_info(crop, country):
    try:
        headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "grok",
            "prompt": f"Provide detailed smart farming guidelines for {crop} in {country}, including fertilizers, time periods, and best practices.",
            "max_tokens": 500
        }
        response = requests.post(XAI_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("text", "📝 No guidance available")
    except Exception as e:
        return f"🚨 Error fetching smart farming guidance: {str(e)}"

def predict_disease(image):
    if disease_model is None:
        return "🛠️ Disease detection model not loaded (placeholder)"
    img = keras_image.img_to_array(image.resize((224, 224))) / 255.0
    img = np.expand_dims(img, axis=0)
    return "🌿 Disease Name (placeholder)"

# Custom CSS with aesthetic background and centered submit buttons
st.markdown(
    """
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background: url('https://images.unsplash.com/photo-1506748686214-e9df14d4d9d0?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80') no-repeat center center fixed;
            background-size: cover;
            color: #2b3e50;
            min-height: 100vh;
        }
        .main {
            padding: 20px;
            max-width: 600px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.85);
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            animation: fadeIn 1s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        h1, h2, h3, h4, h5, h6 {
            text-align: center;
            color: #2b3e50;
            font-weight: bold;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
        }
        .stSelectbox, .stTextInput>div>input, .stNumberInput>div>input {
            background-color: #ffffff;
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 5px;
            color: #2b3e50;
            font-size: 14px;
            width: 100%;
            transition: border-color 0.3s ease;
        }
        .stSelectbox:hover, .stTextInput:hover>div>input, .stNumberInput:hover>div>input {
            border-color: #FF5722;
        }
        .stSelectbox>div>div, .stNumberInput>div>div {
            background-color: #ffffff;
            border-radius: 10px;
        }
        .stButton>button {
            background: linear-gradient(90deg, #4CAF50, #FFEB3B);
            color: white;
            border-radius: 10px;
            padding: 10px;
            border: none;
            width: 200px;
            font-size: 16px;
            font-weight: bold;
            transition: transform 0.3s ease, background 0.3s ease;
            display: block;
            margin: 0 auto;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #FF5722, #FFCA28);
            transform: scale(1.05);
        }
        /* Enhanced Navigation Bar */
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #4CAF50 0%, #2196F3 50%, #FF5722 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            animation: slideIn 1s ease-out;
        }
        @keyframes slideIn {
            from { transform: translateX(-100%); }
            to { transform: translateX(0); }
        }
        .sidebar .sidebar-content .stButton>button {
            background: rgba(255, 255, 255, 0.2);
            color: white;
            margin: 10px 0;
            border-radius: 25px;
            padding: 12px 15px;
            font-size: 16px;
            font-weight: bold;
            display: flex;
            align-items: center;
            justify-content: flex-start;
            transition: transform 0.3s ease, background 0.3s ease, box-shadow 0.3s ease;
            border: 2px solid rgba(255, 255, 255, 0.5);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            width: 100%;
        }
        .sidebar .sidebar-content .stButton>button:hover {
            background: rgba(255, 255, 255, 0.4);
            transform: translateX(10px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .sidebar .sidebar-content .stButton>button::before {
            content: '';
            margin-right: 10px;
        }
        .sidebar .sidebar-content h1 {
            text-align: center;
            font-size: 24px;
            margin-bottom: 20px;
            color: white;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        .card {
            background: linear-gradient(135deg, #ffffff, #e0f7fa);
            border-radius: 10px;
            padding: 15px;
            margin: 10px auto;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            text-align: center;
            max-width: 250px;
            transition: transform 0.3s ease;
        }
        .card:hover {
            transform: translateY(-5px);
        }
        .card h3 {
            color: #FF5722;
            font-size: 18px;
            margin-bottom: 10px;
        }
        .stForm {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            animation: bounceIn 1s ease-out;
        }
        @keyframes bounceIn {
            0% { transform: scale(0.9); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        .stImage {
            border-radius: 10px;
            margin: 10px auto;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            animation: fadeInImage 1s ease-in;
        }
        @keyframes fadeInImage {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state for expenses and profit
if 'expenses' not in st.session_state:
    st.session_state.expenses = []
if 'profit' not in st.session_state:
    st.session_state.profit = []

# User registration session
if 'user_info' not in st.session_state:
    st.title("🌱 Bhoomi - Farmer Registration 📝")
    st.markdown("<p style='text-align: center; color: #FF5722;'>Enter Farmer Details Below 🎉</p>", unsafe_allow_html=True)
    with st.form("user_form", clear_on_submit=True):
        name = st.text_input("👤 Full Name")
        mobile = st.text_input("📞 Mobile Number", help="e.g., 9876543210")
        place = st.text_input("🏡 Place")
        submitted = st.form_submit_button("Submit 🚀")
    if submitted:
        if name and mobile and place:
            st.session_state.user_info = {"name": name, "mobile": mobile, "place": place}
            st.success("✅ Registration successful! Redirecting to dashboard...")
            st.session_state.menu = "Home"
            st.rerun()
        else:
            st.error("🚫 Please fill in all fields.")
else:
    st.title(f"🌱 Bhoomi - Welcome {st.session_state.user_info['name']} 👋")
    st.markdown("<p style='text-align: center; color: #FF5722;'>Your Personalized Farming Dashboard 🌟</p>", unsafe_allow_html=True)

    # Enhanced Sidebar Navigation
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

    # Home page with statistics, expense, and profit input
    if selected_menu == "Home":
        st.subheader("📊 Statistics")
        st.markdown("<p style='text-align: center; color: #FF5722;'>Track Your Finances 🎉</p>", unsafe_allow_html=True)

        # Expense or Profit Input Form
        with st.form("finance_form"):
            finance_type = st.selectbox("📋 Select Type:", ["Expense", "Profit"])
            
            if finance_type == "Expense":
                expense_date = st.date_input("📅 Expense Date", value=datetime.today())
                expense_amount = st.number_input("💸 Expense Amount", min_value=0.0, value=0.0, step=0.1)
                expense_purpose = st.text_input("📝 Expense For")
                submitted = st.form_submit_button("Add Expense 🚀")
                if submitted:
                    if expense_amount >= 0 and expense_purpose:
                        st.session_state.expenses.append({"date": expense_date, "amount": expense_amount, "purpose": expense_purpose})
                        st.success("✅ Expense added successfully!")
                    else:
                        st.error("🚫 Please fill in all fields with valid amounts.")
            else:
                profit_date = st.date_input("📅 Profit Date", value=datetime.today())
                profit_amount = st.number_input("💰 Profit Amount", min_value=0.0, value=0.0, step=0.1)
                submitted = st.form_submit_button("Add Profit 🚀")
                if submitted:
                    if profit_amount >= 0:
                        st.session_state.profit.append({"date": profit_date, "amount": profit_amount})
                        st.success("✅ Profit added successfully!")
                    else:
                        st.error("🚫 Please enter a valid profit amount.")

        # Convert to DataFrames for charting
        df_expenses = pd.DataFrame(st.session_state.expenses)
        df_profit = pd.DataFrame(st.session_state.profit)

        # Aggregate data by date
        if not df_expenses.empty:
            df_expenses = df_expenses.groupby("date").agg({"amount": "sum"}).reset_index()
        if not df_profit.empty:
            df_profit = df_profit.groupby("date").agg({"amount": "sum"}).reset_index()

        # Charts
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("💸 Expenses Over Time")
            if not df_expenses.empty:
                fig_expenses = px.line(df_expenses, x="date", y="amount", title="Expenses", color_discrete_sequence=["#FF5722"])
                st.plotly_chart(fig_expenses, use_container_width=True)
            else:
                st.write("📊 No expense data to display.")

        with col2:
            st.subheader("💰 Profit Over Time")
            if not df_profit.empty:
                fig_profit = px.line(df_profit, x="date", y="amount", title="Profit", color_discrete_sequence=["#4CAF50"])
                st.plotly_chart(fig_profit, use_container_width=True)
            else:
                st.write("📊 No profit data to display.")

        # Combined Chart
        st.subheader("📈 Expenses vs Profit")
        if not df_expenses.empty and not df_profit.empty:
            # Merge data on date, filling missing dates with 0
            df_combined = pd.merge(df_expenses, df_profit, on="date", how="outer", suffixes=('_exp', '_prof')).fillna(0)
            fig_combined = go.Figure()
            fig_combined.add_trace(go.Scatter(x=df_combined["date"], y=df_combined["amount_exp"], mode='lines', name='Expenses', line=dict(color='#FF5722')))
            fig_combined.add_trace(go.Scatter(x=df_combined["date"], y=df_combined["amount_prof"], mode='lines', name='Profit', line=dict(color='#4CAF50')))
            fig_combined.update_layout(title="Expenses vs Profit", xaxis_title="Date", yaxis_title="Amount")
            st.plotly_chart(fig_combined, use_container_width=True)
        else:
            st.write("📊 No data to display for combined chart.")

        # Statistics Cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="card"><h3>📈 Yield Over Time</h3></div>', unsafe_allow_html=True)
            fig = px.line(df_yield, x="Date", y="Yield", title="", color_discrete_sequence=["#FF5722"])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown('<div class="card"><h3>💰 Total Income</h3><p>0</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="card"><h3>🛒 Total Expenses</h3><p>0</p></div>', unsafe_allow_html=True)
        col4, col5 = st.columns(2)
        with col4:
            st.markdown('<div class="card"><h3>🌾 Total Yield</h3><p>0</p></div>', unsafe_allow_html=True)
        with col5:
            st.markdown('<div class="card"><h3>📈 Total Profit</h3><p>0</p></div>', unsafe_allow_html=True)

    elif selected_menu == "Crop Recommendation":
        st.subheader("🌾 Crop Recommendation System")
        st.markdown("<p style='text-align: center; color: #FF5722;'>Enter Soil Details Below 🎉</p>", unsafe_allow_html=True)
        with st.form("crop_form"):
            nitrogen = st.number_input("🌿 Nitrogen Level (N)", min_value=0, value=0, step=1)
            phosphorus = st.number_input("🌱 Phosphorus Level (P)", min_value=0, value=0, step=1)
            potassium = st.number_input("🌿 Potassium Level (K)", min_value=0, value=0, step=1)
            ph = st.number_input("⚗️ pH Level", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
            rainfall = st.number_input("💧 Rainfall (mm)", min_value=0.0, value=0.0, step=0.1)
            submitted = st.form_submit_button("Predict Crop 🌟")
        if submitted and crop_model:
            if all([nitrogen, phosphorus, potassium, rainfall]):
                features = np.array([[nitrogen, phosphorus, potassium, ph, rainfall]])
                with st.spinner("🔍 Analyzing soil data..."):
                    prediction = crop_model.predict(features)
                st.success(f"🌟 Recommended Crop: **{prediction[0]}**")
            else:
                st.error("🚫 Please fill in all fields.")
        elif submitted and not crop_model:
            st.warning("🛠️ Crop recommendation model not available yet.")

    elif selected_menu == "Identify Plant Disease":
        st.subheader("🦠 Plant Disease Identification")
        st.markdown("<p style='text-align: center; color: #FF5722;'>Upload Plant Image Below 📸</p>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("📷 Upload Plant Image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="🌿 Uploaded Image", use_column_width=True)
            with st.spinner("🔍 Analyzing image..."):
                disease = predict_disease(image)
            st.success(f"🌟 Detected Disease: **{disease}**")

    elif selected_menu == "Crop Yield Prediction":
        st.subheader("📊 Crop Yield Prediction")
        st.markdown("<p style='text-align: center; color: #FF5722;'>Enter Crop Details Below 🎉</p>", unsafe_allow_html=True)
        with st.form("yield_form"):
            countries = ["Albania", "India", "Brazil", "USA", "Australia"]
            country = st.selectbox("🌍 Select Country:", countries)
            crops = ["Maize", "Wheat", "Rice", "Soybean", "Barley"]
            crop = st.selectbox("🌾 Select Crop:", crops)
            rainfall = st.number_input("💧 Average Rainfall (mm/year)", min_value=0.0, value=0.0, step=0.1)
            pesticide = st.number_input("🛡️ Pesticide Use (tonnes)", min_value=0.0, value=0.0, step=0.1)
            temperature = st.number_input("🌡️ Average Temperature (°C)", min_value=-50.0, max_value=50.0, value=0.0, step=0.1)
            submitted = st.form_submit_button("Predict Yield 🚀")
        if submitted:
            if yield_model:
                if all([rainfall, pesticide, temperature]):
                    features = np.array([[rainfall, pesticide, temperature]])
                    with st.spinner("🔍 Predicting yield..."):
                        prediction = yield_model.predict(features)
                    st.success(f"🌟 Predicted Yield: **{prediction[0]:.2f} tons**")
                else:
                    st.error("🚫 Please fill in all fields.")
            else:
                st.warning("🛠️ Yield prediction model not available yet. Placeholder output: **5.0 tons**")

    elif selected_menu == "Today's Weather":
        st.subheader("🌤️ Weather Forecast")
        st.markdown("<p style='text-align: center; color: #FF5722;'>Enter Location Details Below 📍</p>", unsafe_allow_html=True)
        with st.form("weather_form"):
            zip_code = st.text_input("📍 Enter ZIP Code", help="e.g., 110001 for Delhi")
            country_code = st.text_input("🌍 Enter Country Code", value="IN", help="e.g., IN for India")
            submitted = st.form_submit_button("Get Weather 🌞")
        if submitted:
            with st.spinner("🔍 Fetching weather data..."):
                weather_data = get_weather(zip_code, country_code)
            if "error" in weather_data:
                st.error(weather_data["error"])
            elif weather_data.get('main'):
                city_name = weather_data.get('name', 'Unknown Location')
                st.markdown(f"<p style='text-align: center;'>📍 <b>Location</b>: {city_name}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'>🌡️ <b>Temperature</b>: {weather_data['main']['temp']}°C</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'>⛅ <b>Weather</b>: {weather_data['weather'][0]['description'].capitalize()}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'>💧 <b>Humidity</b>: {weather_data['main']['humidity']}%</p>", unsafe_allow_html=True)
            else:
                st.error("🚫 Could not retrieve weather data.")

    elif selected_menu == "Fertilizer Recommendation":
        st.subheader("🧪 Fertilizer Recommendation")
        st.markdown("<p style='text-align: center; color: #FF5722;'>Enter Crop & Soil Details Below 🎉</p>", unsafe_allow_html=True)
        with st.form("fertilizer_form"):
            crop = st.text_input("🌾 Enter Crop Name", help="e.g., Rice")
            soil_type = st.text_input("🌍 Enter Soil Type", help="e.g., Sandy")
            submitted = st.form_submit_button("Recommend Fertilizer 🌟")
        if submitted:
            if fertilizer_model:
                if crop and soil_type:
                    features = np.array([[hash(crop) % 100, hash(soil_type) % 100]])
                    with st.spinner("🔍 Analyzing..."):
                        prediction = fertilizer_model.predict(features)
                    st.success(f"🌟 Recommended Fertilizer: **{prediction[0]}**")
                else:
                    st.error("🚫 Please fill in all fields.")
            else:
                st.warning("🛠️ Fertilizer recommendation model not available yet. Placeholder output: **NPK 20-20-20**")

    elif selected_menu == "Smart Farming Guidance":
        st.subheader("📚 Smart Farming Guidance")
        st.markdown("<p style='text-align: center; color: #FF5722;'>Enter Farming Details Below 🎉</p>", unsafe_allow_html=True)
        with st.form("guidance_form"):
            crop = st.text_input("🌾 Enter Crop Name", help="e.g., Wheat")
            country = st.text_input("🌍 Enter Country Name", help="e.g., India")
            submitted = st.form_submit_button("Get Guidance 🚀")
        if submitted:
            if crop and country:
                with st.spinner("🔍 Fetching guidance..."):
                    guidance = get_smart_farming_info(crop, country)
                st.markdown(f"<div style='text-align: center;'>{guidance}</div>", unsafe_allow_html=True)
                st.image(f"https://source.unsplash.com/600x400/?{crop}", caption=f"🌿 {crop.capitalize()}", use_column_width=True)
            else:
                st.error("🚫 Please fill in all fields.")
