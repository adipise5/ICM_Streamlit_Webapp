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

# Load ML models and label encoders with caching and error handling
@st.cache_resource
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"🚨 Model file not found: {model_path}")
        return None

# Load the crop recommendation model (fixed file name)
crop_model = load_model('models/crop_recommendation.pkl')

# Load the fertilizer recommendation model and label encoders
fertilizer_model = load_model('models/fertilizer_recommendation_model.pkl')
label_encoder_soil = load_model('models/label_encoder_soil.pkl')
label_encoder_crop = load_model('models/label_encoder_crop.pkl')

# Placeholder for yield model
yield_model = None

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
    img = np.expand_dims(img, axis=0)
    return "🌿 Disease Name (placeholder)"

# Custom CSS and JavaScript for animated navbar
st.markdown(
    """
    <style>
        /* Sticky Navigation */
        [data-testid="stSidebar"] {
            position: sticky;
            top: 0;
            height: 100vh;
            z-index: 1000;
            transition: transform 0.3s ease, background-color 0.3s ease;
        }

        /* Navbar Container */
        .navbar {
            background: linear-gradient(135deg, #4CAF50, #2E7D32);
            padding: 15px 20px;
            border-radius: 0 0 15px 15px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            animation: fadeIn 0.5s ease-in-out;
        }

        /* Navigation Items */
        .nav-item {
            display: inline-block;
            margin: 0 15px;
            color: white;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: color 0.3s ease, transform 0.3s ease;
            position: relative;
        }

        .nav-item:hover {
            color: #FFD700; /* Gold hover effect */
            transform: scale(1.1);
            animation: slideUp 0.3s ease;
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

# Initialize session state for expenses, profit, and theme
if 'expenses' not in st.session_state:
    st.session_state.expenses = []
if 'profit' not in st.session_state:
    st.session_state.profit = []
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'  # Default theme

# User registration session
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

    # Modern Animated Navigation Bar
    with st.sidebar:
        st.markdown('<h1 style="margin: 0;">🌍 Navigation</h1>', unsafe_allow_html=True)
        
        # Theme Toggle Button
        st.markdown(
            '<div class="theme-toggle" onclick="toggleTheme()">💡</div>',
            unsafe_allow_html=True
        )

        # Navigation Items with Dropdown
        nav_items = {
                "Home": "🏠 Home",
                "Crop Recommendation": "🌾 Crop Recommendation",
                "Identify Plant Disease": "🦠 Identify Plant Disease",
                "Crop Yield Prediction": "📊 Crop Yield Prediction",
                "Today's Weather": "🌤️ Today's Weather",
                "Fertilizer Recommendation": "🧪 Fertilizer Recommendation",
                "Smart Farming Guidance": "📚 Smart Farming Guidance"
        }

        if 'menu' not in st.session_state:
            st.session_state['menu'] = "Home"

        for key, value in nav_items.items():
            if isinstance(value, dict):
                st.markdown(f'<div class="dropdown">', unsafe_allow_html=True)
                st.markdown(f'<span class="nav-item">{value[list(value.keys())[0]]}</span>', unsafe_allow_html=True)
                st.markdown('<div class="dropdown-content">', unsafe_allow_html=True)
                for sub_key, sub_value in value.items():
                    if st.button(sub_value, key=sub_key):
                        st.session_state['menu'] = sub_key
                        st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                if st.button(value, key=key):
                    st.session_state['menu'] = key
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Assign selected_menu from session state
    selected_menu = st.session_state['menu']

    # Page Content
    if selected_menu == "Home":
        st.subheader("📊 Financial Overview")
        st.markdown("<p style='text-align: center; color: #4CAF50;'>Track Your Finances 🎉</p>", unsafe_allow_html=True)

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
        st.markdown("<p style='text-align: center; color: #4CAF50;'>Enter Soil and Climate Details Below 🎉</p>", unsafe_allow_html=True)
        with st.form("crop_form"):
            nitrogen = st.number_input("🌿 Nitrogen (N) (kg/ha)", min_value=0.0, value=0.0, step=0.1)
            phosphorus = st.number_input("🌱 Phosphorus (P) (kg/ha)", min_value=0.0, value=0.0, step=0.1)
            potassium = st.number_input("🌿 Potassium (K) (kg/ha)", min_value=0.0, value=0.0, step=0.1)
            temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
            humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
            ph = st.number_input("⚗️ pH Level", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
            rainfall = st.number_input("☔ Rainfall (mm)", min_value=0.0, value=0.0, step=0.1)
            submitted = st.form_submit_button("Predict Crop 🌟")
        if submitted and crop_model:
            if all([nitrogen >= 0, phosphorus >= 0, potassium >= 0, temperature >= 0, humidity >= 0, ph >= 0, rainfall >= 0]):
                features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
                with st.spinner("🔍 Analyzing soil and climate data..."):
                    prediction = crop_model.predict(features)
                st.success(f"🌟 Recommended Crop: **{prediction[0]}**")
            else:
                st.error("🚫 Please fill in all fields with valid values.")
        elif submitted and not crop_model:
            st.error("🚫 Crop recommendation model failed to load. Please ensure the model file exists.")

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
                if all([rainfall >= 0, pesticide >= 0, temperature >= 0]):
                    features = np.array([[rainfall, pesticide, temperature]])
                    with st.spinner("🔍 Predicting yield..."):
                        prediction = yield_model.predict(features)
                    st.success(f"🌟 Predicted Yield: **{prediction[0]:.2f} tons**")
                else:
                    st.error("🚫 Please fill in all fields with valid values.")
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
            temparature = st.number_input("🌡️ Temparature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
            humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
            moisture = st.number_input("💦 Moisture (%)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
            col1, col2 = st.columns(2)
            with col1:
                soil_type = st.selectbox("🌍 Soil Type", ["Sandy", "Loamy", "Black", "Red", "Clayey"])
            with col2:
                crop_type = st.selectbox("🌾 Crop Type", ["Maize", "Sugarcane", "Cotton", "Tobacco", "Paddy", "Barley", "Wheat", "Millets", "Oil seeds", "Pulses", "Ground Nuts"])
            nitrogen = st.number_input("🌿 Nitrogen (N) (kg/ha)", min_value=0.0, value=0.0, step=0.1)
            potassium = st.number_input("🌿 Potassium (K) (kg/ha)", min_value=0.0, value=0.0, step=0.1)
            phosphorous = st.number_input("🌱 Phosphorous (P) (kg/ha)", min_value=0.0, value=0.0, step=0.1)
            submitted = st.form_submit_button("Recommend Fertilizer 🌟")
        if submitted:
            if fertilizer_model and label_encoder_soil and label_encoder_crop:
                if all([temparature >= 0, humidity >= 0, moisture >= 0, nitrogen >= 0, potassium >= 0, phosphorous >= 0]):
                    soil_encoded = label_encoder_soil.transform([soil_type])[0]
                    crop_encoded = label_encoder_crop.transform([crop_type])[0]
                    features = np.array([[temparature, humidity, moisture, soil_encoded, crop_encoded, nitrogen, potassium, phosphorous]])
                    with st.spinner("🔍 Analyzing..."):
                        prediction = fertilizer_model.predict(features)
                    st.success(f"🌟 Recommended Fertilizer: **{prediction[0]}**")
                else:
                    st.error("🚫 Please fill in all fields with valid values.")
            else:
                st.error("🚫 Fertilizer recommendation model or label encoders failed to load. Please ensure the model files exist.")

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
