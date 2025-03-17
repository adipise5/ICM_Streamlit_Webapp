import streamlit as st
import requests
from PIL import Image
import numpy as np
import joblib
import os
import io
from tensorflow.keras.preprocessing import image as keras_image
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables (if needed)
load_dotenv()

# Must be the first Streamlit command
st.set_page_config(page_title="Bhoomi Dashboard", layout="wide", initial_sidebar_state="expanded")

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

# Static crop information database (sourced from FAO)
CROP_INFO = {
    "wheat": {
        "climate": "Temperate regions, prefers cool and moist weather during vegetative growth, dry and warm weather during grain filling.",
        "soil": "Well-drained loamy soils, pH 6.0–7.5.",
        "fertilizers": "Nitrogen (120–150 kg/ha), Phosphorus (60–80 kg/ha), Potassium (40–60 kg/ha). Apply NPK 20-20-20 at planting, followed by split nitrogen applications.",
        "time_periods": "Sown in autumn (October–November) for winter wheat, spring (March–April) for spring wheat; harvested after 4–5 months.",
        "best_practices": "Rotate with legumes, ensure proper irrigation (500–800 mm rainfall), control weeds early, and use disease-resistant varieties."
    },
    "rice": {
        "climate": "Tropical and subtropical regions, warm and humid, temperatures 20–38°C.",
        "soil": "Clayey or loamy soils with good water retention, pH 5.5–7.0.",
        "fertilizers": "Nitrogen (100–150 kg/ha), Phosphorus (30–50 kg/ha), Potassium (30–50 kg/ha). Apply NPK 15-15-15 at planting, split nitrogen applications during tillering and panicle initiation.",
        "time_periods": "Sown during the monsoon (June–July), harvested after 4–6 months (November–December).",
        "best_practices": "Flooded fields for most varieties (irrigated rice), transplant seedlings at 20–30 days, manage pests like rice blast, and ensure 1000–1500 mm water availability."
    },
    "maize": {
        "climate": "Warm weather, 21–30°C, requires frost-free conditions.",
        "soil": "Well-drained sandy loam to loamy soils, pH 5.8–7.0.",
        "fertilizers": "Nitrogen (120–180 kg/ha), Phosphorus (60–80 kg/ha), Potassium (40–60 kg/ha). Apply NPK 20-20-20 at planting, top-dress with nitrogen at knee-high stage.",
        "time_periods": "Sown in spring (April–May), harvested after 3–4 months (August–September).",
        "best_practices": "Plant in rows with 60–75 cm spacing, irrigate at 600–800 mm, control pests like maize borers, and rotate with legumes to improve soil fertility."
    }
}

@st.cache_data
def get_smart_farming_info(crop, country):
    crop = crop.lower()
    if crop not in CROP_INFO:
        return f"🚫 Sorry, detailed guidance for {crop} is not available in the database. General advice: Use balanced NPK fertilizers (20-20-20), ensure proper irrigation, and plant during the optimal season for your region."
    crop_data = CROP_INFO[crop]
    guidance = (
        f"### Smart Farming Guidance for {crop.capitalize()} in {country}\n\n"
        f"**Climate Requirements**: {crop_data['climate']}\n\n"
        f"**Soil Requirements**: {crop_data['soil']}\n\n"
        f"**Fertilizers**: {crop_data['fertilizers']}\n\n"
        f"**Time Periods**: {crop_data['time_periods']}\n\n"
        f"**Best Practices**: {crop_data['best_practices']}\n\n"
        f"**Note**: Adjust practices based on local conditions in {country}, such as rainfall patterns and temperature variations."
    )
    return guidance

def predict_disease(image):
    if disease_model is None:
        return "🛠️ Disease detection model not loaded (placeholder)"
    img = keras_image.img_to_array(image.resize((224, 224))) / 255.0
    img = np.expand_dims(img, axis=0)
    return "🌿 Disease Name (placeholder)"

# Custom CSS with updated sidebar navigation
st.markdown(
    """
    <style>
        /* Nature-inspired textured background */
        [data-testid="stAppViewContainer"] {
            background-color: #e6e8d5; /* Soft earthy beige */
            background-image: radial-gradient(circle, rgba(76, 175, 80, 0.1) 1px, transparent 1px);
            background-size: 20px 20px; /* Subtle texture */
        }
        /* Main content area */
        [data-testid="stAppViewContainer"] > .main {
            background: rgba(255, 255, 255, 0.95); /* Light mode */
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            color: #3a4f41; /* Earthy green text */
            transition: all 0.3s ease;
        }
        /* Dark mode adjustments */
        @media (prefers-color-scheme: dark) {
            [data-testid="stAppViewContainer"] {
                background-color: #2e2e2e; /* Dark earthy gray */
                background-image: radial-gradient(circle, rgba(76, 175, 80, 0.2) 1px, transparent 1px);
            }
            [data-testid="stAppViewContainer"] > .main {
                background: rgba(40, 44, 52, 0.95); /* Dark mode */
                color: #d4e6d5; /* Light earthy green text */
            }
            h1, h2, h3, h4, h5, h6 {
                color: #d4e6d5;
            }
            .stTextInput > div > input, .stNumberInput > div > input, .stSelectbox > div > div {
                background-color: #555;
                color: #d4e6d5;
                border: 1px solid #4CAF50;
            }
        }
        h1, h2, h3, h4, h5, h6 {
            text-align: center;
            color: #3a4f41; /* Light mode earthy green */
            font-weight: bold;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
        }
        /* Smaller input boxes */
        .stTextInput > div > input, .stNumberInput > div > input {
            width: 150px; /* Reduced width */
            padding: 4px; /* Reduced padding */
            font-size: 12px; /* Smaller font size */
            border-radius: 8px;
            border: 1px solid #4CAF50;
            background-color: #fff; /* Light mode */
            color: #3a4f41;
        }
        .stSelectbox > div > div {
            width: 150px; /* Reduced width */
            padding: 4px; /* Reduced padding */
            font-size: 12px; /* Smaller font size */
            border-radius: 8px;
            border: 1px solid #4CAF50;
            background-color: #fff; /* Light mode */
            color: #3a4f41;
        }
        .stButton>button {
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            color: white;
            border-radius: 10px;
            padding: 8px; /* Slightly smaller padding */
            border: none;
            width: 150px; /* Match input box width */
            font-size: 14px; /* Slightly smaller font */
            font-weight: bold;
            transition: transform 0.3s ease, background 0.3s ease;
            display: block;
            margin: 0 auto;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #388E3C, #689F38);
            transform: scale(1.05);
        }
        /* Smaller and centered forms */
        .stForm {
            background: rgba(255, 255, 255, 0.98); /* Light mode */
            padding: 20px; /* Reduced padding */
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            width: 500px; /* Smaller fixed width */
            margin: 0 auto; /* Center the form */
        }
        @media (prefers-color-scheme: dark) {
            .stForm {
                background: rgba(40, 44, 52, 0.98); /* Dark mode */
            }
        }
        /* Modern Sidebar Navigation */
        [data-testid="stSidebar"] > div:first-child {
            background: linear-gradient(180deg, #4CAF50, #388E3C, #2E7D32);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }
        [data-testid="stSidebar"] h1 {
            color: white;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            margin-bottom: 20px;
        }
        /* Style the selectbox to look modern */
        [data-testid="stSidebar"] .stSelectbox > div > div {
            background: rgba(255, 255, 255, 0.2);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s ease;
            width: 100%;
        }
        [data-testid="stSidebar"] .stSelectbox > div > div:hover {
            background: rgba(255, 255, 255, 0.4);
            cursor: pointer;
        }
        .stImage {
            border-radius: 10px;
            margin: 10px auto;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        }
        /* Table styling */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        tr:hover {
            background-color: #f1f1f1;
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

# User registration session (email removed)
if 'user_info' not in st.session_state:
    st.title("🌱 Bhoomi - Farmer Registration 📝")
    st.markdown("<p style='text-align: center; color: #4CAF50;'>Enter Farmer Details Below 🎉</p>", unsafe_allow_html=True)
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
    st.markdown("<p style='text-align: center; color: #4CAF50;'>Your Personalized Farming Dashboard 🌟</p>", unsafe_allow_html=True)

    # Modern Sidebar Navigation with Dropdown
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
    
    selected_menu = st.sidebar.selectbox(
        "Select a Page",
        list(menu_options.keys()),
        format_func=lambda x: menu_options[x],
        index=list(menu_options.keys()).index(st.session_state['menu'])
    )
    st.session_state['menu'] = selected_menu

    # Home page with tables instead of charts
    if selected_menu == "Home":
        st.subheader("📊 Financial Overview")
        st.markdown("<p style='text-align: center; color: #4CAF50;'>Track Your Finances 🎉</p>", unsafe_allow_html=True)

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
                        st.session_state.expenses.append({"date": expense_date.strftime('%Y-%m-%d'), "amount": expense_amount, "purpose": expense_purpose})
                        st.success("✅ Expense added successfully!")
                        st.rerun()
                    else:
                        st.error("🚫 Please fill in all fields with valid amounts.")
            else:
                profit_date = st.date_input("📅 Profit Date", value=datetime.today())
                profit_amount = st.number_input("💰 Profit Amount", min_value=0.0, value=0.0, step=0.1)
                submitted = st.form_submit_button("Add Profit 🚀")
                if submitted:
                    if profit_amount >= 0:
                        st.session_state.profit.append({"date": profit_date.strftime('%Y-%m-%d'), "amount": profit_amount})
                        st.success("✅ Profit added successfully!")
                        st.rerun()
                    else:
                        st.error("🚫 Please enter a valid profit amount.")

        # Display tables with totals
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💸 Expenses")
            if st.session_state.expenses:
                df_expenses = pd.DataFrame(st.session_state.expenses)
                total_expense = df_expenses['amount'].sum()
                st.table(df_expenses)
                st.markdown(f"**Total Expense:** ₹{total_expense:.2f}")
            else:
                st.write("📊 No expense data to display.")
        
        with col2:
            st.subheader("💰 Profits")
            if st.session_state.profit:
                df_profit = pd.DataFrame(st.session_state.profit)
                total_profit = df_profit['amount'].sum()
                st.table(df_profit)
                st.markdown(f"**Total Profit:** ₹{total_profit:.2f}")
            else:
                st.write("📊 No profit data to display.")

    elif selected_menu == "Crop Recommendation":
        st.subheader("🌾 Crop Recommendation System")
        st.markdown("<p style='text-align: center; color: #4CAF50;'>Enter Soil Details Below 🎉</p>", unsafe_allow_html=True)
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
        st.markdown("<p style='text-align: center; color: #4CAF50;'>Upload Plant Image Below 📸</p>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("📷 Upload Plant Image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="🌿 Uploaded Image", use_column_width=True)
            with st.spinner("🔍 Analyzing image..."):
                disease = predict_disease(image)
            st.success(f"🌟 Detected Disease: **{disease}**")

    elif selected_menu == "Crop Yield Prediction":
        st.subheader("📊 Crop Yield Prediction")
        st.markdown("<p style='text-align: center; color: #4CAF50;'>Enter Crop Details Below 🎉</p>", unsafe_allow_html=True)
        with st.form("yield_form"):
            col1, col2 = st.columns(2)
            with col1:
                countries = ["India", "Brazil", "USA", "Australia", "Albania"]
                country = st.selectbox("🌍 Select Country:", countries)
            with col2:
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
        st.markdown("<p style='text-align: center; color: #4CAF50;'>Enter Location Details Below 📍</p>", unsafe_allow_html=True)
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
        st.markdown("<p style='text-align: center; color: #4CAF50;'>Enter Crop & Soil Details Below 🎉</p>", unsafe_allow_html=True)
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
        st.markdown("<p style='text-align: center; color: #4CAF50;'>Enter Farming Details Below 🎉</p>", unsafe_allow_html=True)
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
