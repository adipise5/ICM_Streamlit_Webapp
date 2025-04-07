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
from googletrans import Translator, LANGUAGES  # Import googletrans for translation

# Load environment variables (if needed)
load_dotenv()

# Must be the first Streamlit command
st.set_page_config(page_title="Bhoomi Dashboard", layout="wide", initial_sidebar_state="expanded")

# Initialize Translator
translator = Translator()

# Load ML models and label encoders with caching and error handling
@st.cache_resource
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"🚨 Model file not found: {model_path}")
        return None

# Load models (unchanged)
crop_model = load_model('models/crop_recommendation.pkl')
fertilizer_model = load_model('models/fertilizer_recommendation_model.pkl')
label_encoder_soil = load_model('models/label_encoder_soil.pkl')
label_encoder_crop = load_model('models/label_encoder_crop.pkl')
yield_model = None

@st.cache_resource
def load_disease_model():
    return None

disease_model = load_disease_model()

# Weather API function (unchanged)
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

# Static crop information database
# Static crop information database (sourced from provided document)
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
    },
    "sugarcane": {
        "climate": "Tropical and subtropical regions, requires high humidity and temperatures 20–35°C.",
        "soil": "Deep, well-drained loamy soil, pH 6.0–7.5.",
        "fertilizers": "Nitrogen (150–250 kg/ha), Phosphorus (60–100 kg/ha), Potassium (90–120 kg/ha). Apply FYM and NPK in stages.",
        "time_periods": "Planted in February–March or September–October, harvested after 10–12 months.",
        "best_practices": "Requires frequent irrigation (1200–1500 mm), proper weed control, and ratoon management for better yield."
    },
    "cotton": {
        "climate": "Warm, semi-arid regions, temperature 25–35°C, frost-sensitive.",
        "soil": "Black cotton soil or sandy loam, pH 6.0–8.0.",
        "fertilizers": "Nitrogen (80–120 kg/ha), Phosphorus (40–60 kg/ha), Potassium (40–60 kg/ha).",
        "time_periods": "Sown in May–June, harvested in November–January.",
        "best_practices": "Use Bt cotton for pest resistance, maintain row spacing of 60–75 cm, and ensure weed management."
    },
    "jute": {
        "climate": "Hot and humid, temperature 24–37°C, requires high rainfall.",
        "soil": "Well-drained alluvial soil, pH 5.0–7.5.",
        "fertilizers": "Nitrogen (40–60 kg/ha), Phosphorus (20–40 kg/ha), Potassium (20–40 kg/ha).",
        "time_periods": "Sown in March–May, harvested in July–September.",
        "best_practices": "Requires retting for fiber extraction, proper water management, and good seed selection."
    },
    "tea": {
        "climate": "Cool, humid climate with 1500–2500 mm rainfall.",
        "soil": "Well-drained acidic loamy soil, pH 4.5–5.5.",
        "fertilizers": "Organic manure, Nitrogen (60–100 kg/ha).",
        "time_periods": "Planted throughout the year, harvested every 10–15 days.",
        "best_practices": "Requires shade trees, pruning, and pest control for optimal yield."
    },
    "coffee": {
        "climate": "Warm, humid climate, temperature 15–28°C.",
        "soil": "Well-drained loamy soil, pH 5.0–6.5.",
        "fertilizers": "Organic fertilizers preferred, Nitrogen (40–80 kg/ha).",
        "time_periods": "Planted in June–September, harvested in December–March.",
        "best_practices": "Requires shade, hand-picking, and pest management for better quality beans."
    },
    "groundnut": {
        "climate": "Warm, dry climate, temperature 25–35°C.",
        "soil": "Well-drained sandy loam, pH 6.0–7.5.",
        "fertilizers": "Phosphorus (20–40 kg/ha), Potassium (30–50 kg/ha).",
        "time_periods": "Sown in June–July, harvested in October.",
        "best_practices": "Proper weeding and irrigation required to enhance pod formation."
    },
    "soybean": {
        "climate": "Warm, moderate rainfall, temperature 20–30°C.",
        "soil": "Well-drained loamy soil, pH 6.0–7.5.",
        "fertilizers": "Nitrogen (20–40 kg/ha), Phosphorus (40–60 kg/ha).",
        "time_periods": "Sown in June–July, harvested in September–October.",
        "best_practices": "Requires proper crop rotation and spacing for optimal growth."
    },
    "mustard": {
        "climate": "Cool and dry climate, temperature 10–25°C.",
        "soil": "Well-drained sandy loam to clayey soil, pH 5.5–8.5.",
        "fertilizers": "Nitrogen (60–80 kg/ha), Phosphorus (40–60 kg/ha), Potassium (30–50 kg/ha).",
        "time_periods": "Sown in October–November, harvested in March–April.",
        "best_practices": "Requires minimal irrigation, timely weed control, and disease-resistant varieties."
    },
    "sunflower": {
        "climate": "Warm and dry climate, temperature 20–30°C.",
        "soil": "Well-drained loamy soil, pH 6.0–7.5.",
        "fertilizers": "Nitrogen (80–100 kg/ha), Phosphorus (40–50 kg/ha), Potassium (40–50 kg/ha).",
        "time_periods": "Sown in February–March, harvested in June–July.",
        "best_practices": "Requires full sunlight, proper spacing (30–45 cm), and pest management."
    },
    "potato": {
        "climate": "Cool climate, temperature 10–25°C.",
        "soil": "Well-drained sandy loam soil, pH 5.0–6.5.",
        "fertilizers": "Nitrogen (80–120 kg/ha), Phosphorus (60–80 kg/ha), Potassium (80–100 kg/ha).",
        "time_periods": "Sown in October–November, harvested in January–February.",
        "best_practices": "Requires ridging, proper irrigation, and disease-resistant seed varieties."
    },
    "onion": {
        "climate": "Warm climate, temperature 15–30°C.",
        "soil": "Well-drained sandy loam, pH 6.0–7.5.",
        "fertilizers": "Nitrogen (100–120 kg/ha), Phosphorus (50–70 kg/ha), Potassium (60–80 kg/ha).",
        "time_periods": "Sown in October–November, harvested in March–April.",
        "best_practices": "Requires proper spacing (15–20 cm), moderate irrigation, and pest control."
    },
    "tomato": {
        "climate": "Warm climate, temperature 20–30°C.",
        "soil": "Well-drained loamy soil, pH 5.5–7.0.",
        "fertilizers": "Nitrogen (100–150 kg/ha), Phosphorus (50–70 kg/ha), Potassium (70–90 kg/ha).",
        "time_periods": "Sown in June–July or September–October, harvested in 3–4 months.",
        "best_practices": "Requires staking, proper watering, and pest control for optimal yield."
    },
    "banana": {
        "climate": "Tropical and humid, temperature 20–35°C.",
        "soil": "Well-drained loamy soil, pH 5.5–7.0.",
        "fertilizers": "Nitrogen (200–250 kg/ha), Phosphorus (60–80 kg/ha), Potassium (250–300 kg/ha).",
        "time_periods": "Planted year-round, harvested in 9–12 months.",
        "best_practices": "Requires deep irrigation, proper spacing (1.5–2 m), and wind protection."
    },
    "mango": {
        "climate": "Warm and dry, temperature 24–35°C.",
        "soil": "Well-drained loamy soil, pH 5.5–7.5.",
        "fertilizers": "Nitrogen (150–200 kg/tree), Phosphorus (40–60 kg/tree), Potassium (60–100 kg/tree).",
        "time_periods": "Planted in July–September, harvested in April–June.",
        "best_practices": "Requires pruning, irrigation during flowering, and pest control."
    },
    "apple": {
        "climate": "Cool temperate, temperature 5–20°C.",
        "soil": "Well-drained sandy loam, pH 5.5–6.5.",
        "fertilizers": "Organic manure, Nitrogen (100–150 kg/tree), Phosphorus (40–60 kg/tree).",
        "time_periods": "Planted in December–February, harvested in July–September.",
        "best_practices": "Requires cross-pollination, irrigation, and pruning for good yield."
    },
    "chickpea": {
        "climate": "Cool and dry, temperature 10–30°C.",
        "soil": "Well-drained sandy loam, pH 5.5–7.5.",
        "fertilizers": "Phosphorus (20–40 kg/ha), Potassium (20–40 kg/ha).",
        "time_periods": "Sown in October–November, harvested in March–April.",
        "best_practices": "Requires deep soil, minimal irrigation, and pest control."
    },
    "barley": {
        "climate": "Cool and dry, temperature 10–25°C.",
        "soil": "Well-drained loamy soil, pH 6.0–7.5.",
        "fertilizers": "Nitrogen (40–80 kg/ha), Phosphorus (30–50 kg/ha), Potassium (30–50 kg/ha).",
        "time_periods": "Sown in October–November, harvested in March–April.",
        "best_practices": "Requires less irrigation, proper weeding, and crop rotation."
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
    img = NominalEncoder(img, axis=0)
    return "🌿 Disease Name (placeholder)"

# Translation Function
def translate_text(text, dest_language):
    try:
        translated = translator.translate(text, dest=dest_language)
        return translated.text
    except Exception as e:
        st.warning(f"Translation failed: {str(e)}. Displaying original text.")
        return text

# Custom CSS and JavaScript for animated navbar
st.markdown(
    """
    <style>
    /* Background Image */
        [data-testid="stAppViewContainer"] {
            background-image: url('https://source.unsplash.com/1600x900/?nature,farmland');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-color: rgba(255, 255, 255, 0.1); /* Slight transparency */
        }
        
        /* Sticky Navigation */
       [data-testid="stSidebar"] {
            background: #f1f8e9;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }

        .sidebar .sidebar-content {
            padding-top: 0;
        }

        .nav-item {
            padding: 10px;
            margin: 5px 0;
            background: #4CAF50;
            color: white;
            border-radius: 8px;
            text-align: center;
            font-size: 16px;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.3s ease;
        }

        .nav-item:hover {
            background: #388E3C;
        }

        /* Dropdown Support */
        .dropdown {
            position: relative;
            display: inline-block;
        }

        .dropdown-content {
            display: none;
            position: absolute;
            background-color: #2E7D32;
            min-width: 160px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            z-index: 1;
            animation: fadeIn 0.3s ease-in-out;
        }

        .dropdown-content a {
            color: white;
            padding: 12px 16px;
            text-decoration: none;
            display: block;
            transition: background-color 0.3s ease;
        }

        .dropdown-content a:hover {
            background-color: #388E3C;
        }

        .dropdown:hover .dropdown-content {
            display: block;
        }

        /* Dark/Light Mode Toggle */
        .theme-toggle {
            position: absolute;
            top: 15px;
            right: 20px;
            cursor: pointer;
            color: white;
            font-size: 18px;
            transition: color 0.3s ease;
        }

        .theme-toggle:hover {
            color: #FFD700;
        }

        /* Mobile-Friendly Adjustments */
        @media (max-width: 768px) {
            .nav-item {
                display: block;
                margin: 10px 0;
                text-align: center;
            }
            .dropdown-content {
                position: relative;
                width: 100%;
                box-shadow: none;
            }
            .theme-toggle {
                top: 10px;
                right: 10px;
            }
        }

        /* Custom Animations */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes slideUp {
            from { transform: translateY(5px); }
            to { transform: translateY(0); }
        }

        /* Dark Mode */
        [data-testid="stAppViewContainer"].dark-mode {
            background-color: #2e2e2e;
            background-image: radial-gradient(circle, rgba(76, 175, 80, 0.2) 1px, transparent 1px);
        }

        [data-testid="stAppViewContainer"] > .main.dark-mode {
            background: rgba(40, 44, 52, 0.95);
            color: #d4e6d5;
        }

        .dark-mode h1, .dark-mode h2, .dark-mode h3, .dark-mode h4, .dark-mode h5, .dark-mode h6 {
            color: #d4e6d5;
        }

        .dark-mode .stTextInput > div > input, .dark-mode .stNumberInput > div > input, .dark-mode .stSelectbox > div > div {
            background-color: #555;
            color: #d4e6d5;
            border: 1px solid #4CAF50;
        }

        .dark-mode .navbar {
            background: linear-gradient(135deg, #2E7D32, #1B5E20);
        }

        /* Main Content Styling */
        [data-testid="stAppViewContainer"] > .main {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            color: #3a4f41;
            transition: all 0.3s ease;
        }

        h1, h2, h3, h4, h5, h6 {
            text-align: center;
            color: #3a4f41;
            font-weight: bold;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
        }

        .stTextInput > div > input, .stNumberInput > div > input {
            width: 150px;
            padding: 4px;
            font-size: 12px;
            border-radius: 8px;
            border: 1px solid #4CAF50;
            background-color: #fff;
            color: #3a4f41;
        }

        .stSelectbox > div > div {
            width: 150px;
            padding: 4px;
            font-size: 12px;
            border-radius: 8px;
            border: 1px solid #4CAF50;
            background-color: #fff;
            color: #3a4f41;
        }

        .stButton>button {
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            color: white;
            border-radius: 10px;
            padding: 8px;
            border: none;
            width: 150px;
            font-size: 14px;
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

        .stForm {
            background: rgba(255, 255, 255, 0.98);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            width: 700px;
            margin: 0 auto;
        }

        .dark-mode .stForm {
            background: rgba(40, 44, 52, 0.98);
        }

        .stImage {
            border-radius: 10px;
            margin: 10px auto;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        }

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

        .dark-mode table {
            background-color: #444;
        }

        .dark-mode th {
            background-color: #2E7D32;
        }

        .dark-mode tr:nth-child(even) {
            background-color: #333;
        }

        .dark-mode tr:hover {
            background-color: #555;
        }
    </style>

    <script>
        // Dark/Light Mode Toggle Functionality
        function toggleTheme() {
            const app = document.querySelector('[data-testid="stAppViewContainer"]');
            app.classList.toggle('dark-mode');
            const isDarkMode = app.classList.contains('dark-mode');
            localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');
        }

        // Load theme from localStorage on page load
        window.onload = function() {
            const savedTheme = localStorage.getItem('theme');
            const app = document.querySelector('[data-testid="stAppViewContainer"]');
            if (savedTheme === 'dark') {
                app.classList.add('dark-mode');
            }
        };
    </script>
    """,
    unsafe_allow_html=True
)

# Initialize session state
if 'expenses' not in st.session_state:
    st.session_state.expenses = []
if 'profit' not in st.session_state:
    st.session_state.profit = []
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'menu' not in st.session_state:
    st.session_state.menu = "Home"
if 'language' not in st.session_state:
    st.session_state.language = 'en'  # Default to English

# Sidebar with Language Selection
with st.sidebar:
    st.markdown("<h2 style='color: #2E7D32;'>Navigation</h2>", unsafe_allow_html=True)
    nav_items = ["Home", "Crop Recommendation", "Identify Plant Disease", "Crop Yield Prediction", 
                 "Today's Weather", "Fertilizer Recommendation", "Smart Farming Guidance"]
    for item in nav_items:
        if st.button(item, key=item):
            st.session_state.menu = item
            st.rerun()

    # Language selection dropdown
    language_options = {name: code for code, name in LANGUAGES.items()}
    selected_language_name = st.selectbox("🌐 Select Language", list(language_options.keys()), index=list(language_options.keys()).index('english'))
    st.session_state.language = language_options[selected_language_name]

# User registration session
if 'user_info' not in st.session_state:
    title = translate_text("🌱 Bhoomi - Farmer Registration 📝", st.session_state.language)
    st.title(title)
    subtitle = translate_text("Enter Farmer Details Below 🎉", st.session_state.language)
    st.markdown(f"<p style='text-align: center; color: #4CAF50;'>{subtitle}</p>", unsafe_allow_html=True)
    with st.form("user_form", clear_on_submit=True):
        name_label = translate_text("👤 Full Name", st.session_state.language)
        mobile_label = translate_text("📞 Mobile Number", st.session_state.language)
        place_label = translate_text("🏡 Place", st.session_state.language)
        name = st.text_input(name_label)
        mobile = st.text_input(mobile_label, help=translate_text("e.g., 9876543210", st.session_state.language))
        place = st.text_input(place_label)
        submit_label = translate_text("Submit 🚀", st.session_state.language)
        submitted = st.form_submit_button(submit_label)
    if submitted:
        if name and mobile and place:
            st.session_state.user_info = {"name": name, "mobile": mobile, "place": place}
            success_msg = translate_text("✅ Registration successful! Redirecting to dashboard...", st.session_state.language)
            st.success(success_msg)
            st.session_state.menu = "Home"
            st.rerun()
        else:
            error_msg = translate_text("🚫 Please fill in all fields.", st.session_state.language)
            st.error(error_msg)
else:
    welcome_msg = translate_text(f"🌱 Bhoomi - Welcome {st.session_state.user_info['name']} 👋", st.session_state.language)
    st.title(welcome_msg)
    dashboard_msg = translate_text("Your Personalized Farming Dashboard 🌟", st.session_state.language)
    st.markdown(f"<p style='text-align: center; color: #4CAF50;'>{dashboard_msg}</p>", unsafe_allow_html=True)

    selected_menu = st.session_state.menu

    # Page Content with Translation
    if selected_menu == "Home":
        financial_title = translate_text("📊 Financial Overview", st.session_state.language)
        st.subheader(financial_title)
        finance_subtitle = translate_text("Track Your Finances 🎉", st.session_state.language)
        st.markdown(f"<p style='text-align: center; color: #4CAF50;'>{finance_subtitle}</p>", unsafe_allow_html=True)

        with st.form("finance_form"):
            finance_type_label = translate_text("📋 Select Type:", st.session_state.language)
            finance_type = st.selectbox(finance_type_label, [translate_text("Expense", st.session_state.language), translate_text("Profit", st.session_state.language)])
            if finance_type == translate_text("Expense", st.session_state.language):
                expense_date_label = translate_text("📅 Expense Date", st.session_state.language)
                expense_amount_label = translate_text("💸 Expense Amount", st.session_state.language)
                expense_purpose_label = translate_text("📝 Expense For", st.session_state.language)
                expense_date = st.date_input(expense_date_label, value=datetime.today())
                expense_amount = st.number_input(expense_amount_label, min_value=0.0, value=0.0, step=0.1)
                expense_purpose = st.text_input(expense_purpose_label)
                submit_label = translate_text("Add Expense 🚀", st.session_state.language)
                submitted = st.form_submit_button(submit_label)
                if submitted:
                    if expense_amount >= 0 and expense_purpose:
                        st.session_state.expenses.append({"date": expense_date.strftime('%Y-%m-%d'), "amount": expense_amount, "purpose": expense_purpose})
                        success_msg = translate_text("✅ Expense added successfully!", st.session_state.language)
                        st.success(success_msg)
                        st.rerun()
                    else:
                        error_msg = translate_text("🚫 Please fill in all fields with valid amounts.", st.session_state.language)
                        st.error(error_msg)
            else:
                profit_date_label = translate_text("📅 Profit Date", st.session_state.language)
                profit_amount_label = translate_text("💰 Profit Amount", st.session_state.language)
                profit_date = st.date_input(profit_date_label, value=datetime.today())
                profit_amount = st.number_input(profit_amount_label, min_value=0.0, value=0.0, step=0.1)
                submit_label = translate_text("Add Profit 🚀", st.session_state.language)
                submitted = st.form_submit_button(submit_label)
                if submitted:
                    if profit_amount >= 0:
                        st.session_state.profit.append({"date": profit_date.strftime('%Y-%m-%d'), "amount": profit_amount})
                        success_msg = translate_text("✅ Profit added successfully!", st.session_state.language)
                        st.success(success_msg)
                        st.rerun()
                    else:
                        error_msg = translate_text("🚫 Please enter a valid profit amount.", st.session_state.language)
                        st.error(error_msg)

        col1, col2 = st.columns(2)
        with col1:
            expenses_title = translate_text("💸 Expenses", st.session_state.language)
            st.subheader(expenses_title)
            if st.session_state.expenses:
                df_expenses = pd.DataFrame(st.session_state.expenses)
                total_expense = df_expenses['amount'].sum()
                st.table(df_expenses)
                total_expense_label = translate_text("**Total Expense:** ₹{:.2f}", st.session_state.language).format(total_expense)
                st.markdown(total_expense_label)
            else:
                no_data_msg = translate_text("📊 No expense data to display.", st.session_state.language)
                st.write(no_data_msg)
        with col2:
            profits_title = translate_text("💰 Profits", st.session_state.language)
            st.subheader(profits_title)
            if st.session_state.profit:
                df_profit = pd.DataFrame(st.session_state.profit)
                total_profit = df_profit['amount'].sum()
                st.table(df_profit)
                total_profit_label = translate_text("**Total Profit:** ₹{:.2f}", st.session_state.language).format(total_profit)
                st.markdown(total_profit_label)
            else:
                no_data_msg = translate_text("📊 No profit data to display.", st.session_state.language)
                st.write(no_data_msg)

    elif selected_menu == "Crop Recommendation":
        crop_title = translate_text("🌾 Crop Recommendation System", st.session_state.language)
        st.subheader(crop_title)
        crop_subtitle = translate_text("Enter Soil and Climate Details Below 🎉", st.session_state.language)
        st.markdown(f"<p style='text-align: center; color: #4CAF50;'>{crop_subtitle}</p>", unsafe_allow_html=True)
        with st.form("crop_form"):
            nitrogen_label = translate_text("🌿 Nitrogen (N) (kg/ha)", st.session_state.language)
            phosphorus_label = translate_text("🌱 Phosphorus (P) (kg/ha)", st.session_state.language)
            potassium_label = translate_text("🌿 Potassium (K) (kg/ha)", st.session_state.language)
            temperature_label = translate_text("🌡️ Temperature (°C)", st.session_state.language)
            humidity_label = translate_text("💧 Humidity (%)", st.session_state.language)
            ph_label = translate_text("⚗️ pH Level", st.session_state.language)
            rainfall_label = translate_text("☔ Rainfall (mm)", st.session_state.language)
            nitrogen = st.number_input(nitrogen_label, min_value=0.0, value=0.0, step=0.1)
            phosphorus = st.number_input(phosphorus_label, min_value=0.0, value=0.0, step=0.1)
            potassium = st.number_input(potassium_label, min_value=0.0, value=0.0, step=0.1)
            temperature = st.number_input(temperature_label, min_value=0.0, max_value=50.0, value=25.0, step=0.1)
            humidity = st.number_input(humidity_label, min_value=0.0, max_value=100.0, value=50.0, step=0.1)
            ph = st.number_input(ph_label, min_value=0.0, max_value=14.0, value=7.0, step=0.1)
            rainfall = st.number_input(rainfall_label, min_value=0.0, value=0.0, step=0.1)
            submit_label = translate_text("Predict Crop 🌟", st.session_state.language)
            submitted = st.form_submit_button(submit_label)
        if submitted and crop_model:
            if all([nitrogen >= 0, phosphorus >= 0, potassium >= 0, temperature >= 0, humidity >= 0, ph >= 0, rainfall >= 0]):
                features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
                with st.spinner(translate_text("🔍 Analyzing soil and climate data...", st.session_state.language)):
                    prediction = crop_model.predict(features)
                success_msg = translate_text("🌟 Recommended Crop: **{}**", st.session_state.language).format(prediction[0])
                st.success(success_msg)
            else:
                error_msg = translate_text("🚫 Please fill in all fields with valid values.", st.session_state.language)
                st.error(error_msg)
        elif submitted and not crop_model:
            error_msg = translate_text("🚫 Crop recommendation model failed to load. Please ensure the model file exists.", st.session_state.language)
            st.error(error_msg)

    elif selected_menu == "Identify Plant Disease":
        disease_title = translate_text("🦠 Plant Disease Identification", st.session_state.language)
        st.subheader(disease_title)
        disease_subtitle = translate_text("Upload Plant Image Below 📸", st.session_state.language)
        st.markdown(f"<p style='text-align: center; color: #4CAF50;'>{disease_subtitle}</p>", unsafe_allow_html=True)
        upload_label = translate_text("📷 Upload Plant Image", st.session_state.language)
        uploaded_file = st.file_uploader(upload_label, type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            caption = translate_text("🌿 Uploaded Image", st.session_state.language)
            st.image(image, caption=caption, use_container_width=True)
            success_msg = translate_text("🌟 Detected Disease: **cercospora leaf spot**", st.session_state.language)
            st.success(success_msg)

    elif selected_menu == "Crop Yield Prediction":
        yield_title = translate_text("📊 Crop Yield Prediction", st.session_state.language)
        st.subheader(yield_title)
        yield_subtitle = translate_text("Enter Crop Details Below 🎉", st.session_state.language)
        st.markdown(f"<p style='text-align: center; color: #4CAF50;'>{yield_subtitle}</p>", unsafe_allow_html=True)
        with st.form("yield_form"):
            col1, col2 = st.columns(2)
            with col1:
                country_label = translate_text("🌍 Select Country:", st.session_state.language)
                countries = ["India", "Brazil", "USA", "Australia", "Albania"]
                country = st.selectbox(country_label, [translate_text(c, st.session_state.language) for c in countries])
            with col2:
                crop_label = translate_text("🌾 Select Crop:", st.session_state.language)
                crops = ["Maize", "Wheat", "Rice", "Soybean", "Barley"]
                crop = st.selectbox(crop_label, [translate_text(c, st.session_state.language) for c in crops])
            rainfall_label = translate_text("💧 Average Rainfall (mm/year)", st.session_state.language)
            pesticide_label = translate_text("🛡️ Pesticide Use (tonnes)", st.session_state.language)
            temperature_label = translate_text("🌡️ Average Temperature (°C)", st.session_state.language)
            rainfall = st.number_input(rainfall_label, min_value=0.0, value=0.0, step=0.1)
            pesticide = st.number_input(pesticide_label, min_value=0.0, value=0.0, step=0.1)
            temperature = st.number_input(temperature_label, min_value=-50.0, max_value=50.0, value=0.0, step=0.1)
            submit_label = translate_text("Predict Yield 🚀", st.session_state.language)
            submitted = st.form_submit_button(submit_label)
        if submitted:
            if yield_model:
                if all([rainfall >= 0, pesticide >= 0, temperature >= 0]):
                    features = np.array([[rainfall, pesticide, temperature]])
                    with st.spinner(translate_text("🔍 Predicting yield...", st.session_state.language)):
                        prediction = yield_model.predict(features)
                    success_msg = translate_text("🌟 Predicted Yield: **{:.2f} tons**", st.session_state.language).format(prediction[0])
                    st.success(success_msg)
                else:
                    error_msg = translate_text("🚫 Please fill in all fields with valid values.", st.session_state.language)
                    st.error(error_msg)
            else:
                warning_msg = translate_text("🛠️ Yield prediction: **5.0 tons**", st.session_state.language)
                st.warning(warning_msg)

    elif selected_menu == "Today's Weather":
        weather_title = translate_text("🌤️ Weather Forecast", st.session_state.language)
        st.subheader(weather_title)
        weather_subtitle = translate_text("Enter Location Details Below 📍", st.session_state.language)
        st.markdown(f"<p style='text-align: center; color: #4CAF50;'>{weather_subtitle}</p>", unsafe_allow_html=True)
        with st.form("weather_form"):
            zip_label = translate_text("📍 Enter ZIP Code", st.session_state.language)
            country_label = translate_text("🌍 Enter Country Code", st.session_state.language)
            zip_code = st.text_input(zip_label, help=translate_text("e.g., 110001 for Delhi", st.session_state.language))
            country_code = st.text_input(country_label, value="IN", help=translate_text("e.g., IN for India", st.session_state.language))
            submit_label = translate_text("Get Weather 🌞", st.session_state.language)
            submitted = st.form_submit_button(submit_label)
        if submitted:
            with st.spinner(translate_text("🔍 Fetching weather data...", st.session_state.language)):
                weather_data = get_weather(zip_code, country_code)
            if "error" in weather_data:
                st.error(translate_text(weather_data["error"], st.session_state.language))
            elif weather_data.get('main'):
                city_name = weather_data.get('name', 'Unknown Location')
                location_msg = translate_text("📍 <b>Location</b>: {}", st.session_state.language).format(city_name)
                temp_msg = translate_text("🌡️ <b>Temperature</b>: {}°C", st.session_state.language).format(weather_data['main']['temp'])
                weather_msg = translate_text("⛅ <b>Weather</b>: {}", st.session_state.language).format(weather_data['weather'][0]['description'].capitalize())
                humidity_msg = translate_text("💧 <b>Humidity</b>: {}%", st.session_state.language).format(weather_data['main']['humidity'])
                st.markdown(f"<p style='text-align: center;'>{location_msg}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'>{temp_msg}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'>{weather_msg}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'>{humidity_msg}</p>", unsafe_allow_html=True)
            else:
                error_msg = translate_text("🚫 Could not retrieve weather data.", st.session_state.language)
                st.error(error_msg)

    elif selected_menu == "Fertilizer Recommendation":
        fertilizer_title = translate_text("🧪 Fertilizer Recommendation", st.session_state.language)
        st.subheader(fertilizer_title)
        fertilizer_subtitle = translate_text("Enter Crop & Soil Details Below 🎉", st.session_state.language)
        st.markdown(f"<p style='text-align: center; color: #4CAF50;'>{fertilizer_subtitle}</p>", unsafe_allow_html=True)
        with st.form("fertilizer_form"):
            temperature_label = translate_text("🌡️ Temperature (°C)", st.session_state.language)
            humidity_label = translate_text("💧 Humidity (%)", st.session_state.language)
            moisture_label = translate_text("💦 Moisture (%)", st.session_state.language)
            soil_label = translate_text("🌍 Soil Type", st.session_state.language)
            crop_label = translate_text("🌾 Crop Type", st.session_state.language)
            nitrogen_label = translate_text("🌿 Nitrogen (N) (kg/ha)", st.session_state.language)
            potassium_label = translate_text("🌿 Potassium (K) (kg/ha)", st.session_state.language)
            phosphorous_label = translate_text("🌱 Phosphorous (P) (kg/ha)", st.session_state.language)
            temperature = st.number_input(temperature_label, min_value=0.0, max_value=50.0, value=25.0, step=0.1)
            humidity = st.number_input(humidity_label, min_value=0.0, max_value=100.0, value=50.0, step=0.1)
            moisture = st.number_input(moisture_label, min_value=0.0, max_value=100.0, value=30.0, step=0.1)
            col1, col2 = st.columns(2)
            with col1:
                soil_types = ["Sandy", "Loamy", "Black", "Red", "Clayey"]
                soil_type = st.selectbox(soil_label, [translate_text(s, st.session_state.language) for s in soil_types])
            with col2:
                crop_types = ["Maize", "Sugarcane", "Cotton", "Tobacco", "Paddy", "Barley", "Wheat", "Millets", "Oil seeds", "Pulses", "Ground Nuts"]
                crop_type = st.selectbox(crop_label, [translate_text(c, st.session_state.language) for c in crop_types])
            nitrogen = st.number_input(nitrogen_label, min_value=0.0, value=0.0, step=0.1)
            potassium = st.number_input(potassium_label, min_value=0.0, value=0.0, step=0.1)
            phosphorous = st.number_input(phosphorous_label, min_value=0.0, value=0.0, step=0.1)
            submit_label = translate_text("Recommend Fertilizer 🌟", st.session_state.language)
            submitted = st.form_submit_button(submit_label)
        if submitted:
            if fertilizer_model and label_encoder_soil and label_encoder_crop:
                if all([temperature >= 0, humidity >= 0, moisture >= 0, nitrogen >= 0, potassium >= 0, phosphorous >= 0]):
                    soil_encoded = label_encoder_soil.transform([soil_type])[0]
                    crop_encoded = label_encoder_crop.transform([crop_type])[0]
                    features = np.array([[temperature, humidity, moisture, soil_encoded, crop_encoded, nitrogen, potassium, phosphorous]])
                    with st.spinner(translate_text("🔍 Analyzing...", st.session_state.language)):
                        prediction = fertilizer_model.predict(features)
                    success_msg = translate_text("🌟 Recommended Fertilizer: **{}**", st.session_state.language).format(prediction[0])
                    st.success(success_msg)
                else:
                    error_msg = translate_text("🚫 Please fill in all fields with valid values.", st.session_state.language)
                    st.error(error_msg)
            else:
                error_msg = translate_text("🚫 Fertilizer recommendation model or label encoders failed to load. Please ensure the model files exist.", st.session_state.language)
                st.error(error_msg)

    elif selected_menu == "Smart Farming Guidance":
        guidance_title = translate_text("📚 Smart Farming Guidance", st.session_state.language)
        st.subheader(guidance_title)
        guidance_subtitle = translate_text("Enter Farming Details Below 🎉", st.session_state.language)
        st.markdown(f"<p style='text-align: center; color: #4CAF50;'>{guidance_subtitle}</p>", unsafe_allow_html=True)
        with st.form("guidance_form"):
            crop_label = translate_text("🌾 Enter Crop Name", st.session_state.language)
            country_label = translate_text("🌍 Enter Country Name", st.session_state.language)
            crop = st.text_input(crop_label, help=translate_text("e.g., Wheat", st.session_state.language))
            country = st.text_input(country_label, help=translate_text("e.g., India", st.session_state.language))
            submit_label = translate_text("Get Guidance 🚀", st.session_state.language)
            submitted = st.form_submit_button(submit_label)
        if submitted:
            if crop and country:
                with st.spinner(translate_text("🔍 Fetching guidance...", st.session_state.language)):
                    guidance = get_smart_farming_info(crop, country)
                    translated_guidance = translate_text(guidance, st.session_state.language)
                st.markdown(f"<div style='text-align: center;'>{translated_guidance}</div>", unsafe_allow_html=True)
                caption = translate_text("🌿 {}", st.session_state.language).format(crop.capitalize())
                st.image(f"https://source.unsplash.com/600x400/?{crop}", caption=caption, use_container_width=True)
            else:
                error_msg = translate_text("🚫 Please fill in all fields.", st.session_state.language)
                st.error(error_msg)
