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

crop_model = load_model('models/crop_recommendation.pkl')
fertilizer_model = load_model('models/fertilizer_recommendation_model.pkl')
label_encoder_soil = load_model('models/label_encoder_soil.pkl')
label_encoder_crop = load_model('models/label_encoder_crop.pkl')
yield_model = None

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

# Static crop information database (simplified for brevity)
CROP_INFO = {
    "wheat": {
        "climate": "Temperate regions, cool and moist.",
        "soil": "Well-drained loamy soils, pH 6.0–7.5.",
        "fertilizers": "Nitrogen (120–150 kg/ha), Phosphorus (60–80 kg/ha).",
        "time_periods": "Sown in October–November, harvested after 4–5 months.",
        "best_practices": "Rotate with legumes, proper irrigation."
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

crop_model = load_model('models/crop_recommendation.pkl')
fertilizer_model = load_model('models/fertilizer_recommendation_model.pkl')
label_encoder_soil = load_model('models/label_encoder_soil.pkl')
label_encoder_crop = load_model('models/label_encoder_crop.pkl')
yield_model = None

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

# Static crop information database (simplified for brevity)
CROP_INFO = {
    "wheat": {
        "climate": "Temperate regions, cool and moist.",
        "soil": "Well-drained loamy soils, pH 6.0–7.5.",
        "fertilizers": "Nitrogen (120–150 kg/ha), Phosphorus (60–80 kg/ha).",
        "time_periods": "Sown in October–November, harvested after 4–5 months.",
        "best_practices": "Rotate with legumes, proper irrigation."
    },
    # Add other crops as needed
}

@st.cache_data
def get_smart_farming_info(crop, country):
    crop = crop.lower()
    if crop not in CROP_INFO:
        return f"🚫 No detailed guidance for {crop}. Use balanced NPK fertilizers."
    crop_data = CROP_INFO[crop]
    guidance = (
        f"### Guidance for {crop.capitalize()} in {country}\n\n"
        f"**Climate**: {crop_data['climate']}\n\n"
        f"**Soil**: {crop_data['soil']}\n\n"
        f"**Fertilizers**: {crop_data['fertilizers']}\n\n"
        f"**Time Periods**: {crop_data['time_periods']}\n\n"
        f"**Best Practices**: {crop_data['best_practices']}"
    )
    return guidance

def predict_disease(image):
    if disease_model is None:
        return "🛠️ Disease detection model not loaded (placeholder)"
    img = keras_image.img_to_array(image.resize((224, 224))) / 255.0
    img = np.expand_dims(img, axis=0)
    return "🌿 Disease Name (placeholder)"

# Custom CSS and JavaScript
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

        /* Content Box */
        .content-box {
            background: rgba(255, 255, 255, 0.9);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            margin: 20px auto;
            max-width: 900px;
        }

        /* Simplified Sidebar */
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

        /* Pop-up Notification */
        .notification {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #4CAF50;
            color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            z-index: 1000;
            text-align: center;
            animation: fadeIn 0.5s ease-in-out;
            display: none;
        }

        .notification span {
            font-weight: bold;
            background: #FFD700;
            color: #333;
            padding: 5px 10px;
            border-radius: 5px;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        /* Form Adjustments */
        .stForm {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: space-between;
        }

        .stTextInput, .stNumberInput, .stSelectbox {
            flex: 1 1 45%;
            min-width: 150px;
        }

        .stButton>button {
            background: #4CAF50;
            color: white;
            border-radius: 10px;
            padding: 8px 16px;
            border: none;
            font-weight: bold;
            transition: background 0.3s ease;
            margin: 10px auto;
            display: block;
        }

        .stButton>button:hover {
            background: #388E3C;
        }

        h1, h2, h3 {
            color: #2E7D32;
            text-align: center;
        }
    </style>

    <script>
        function showNotification(message) {
            let notification = document.createElement('div');
            notification.className = 'notification';
            notification.innerHTML = message;
            document.body.appendChild(notification);
            notification.style.display = 'block';
            setTimeout(() => {
                notification.style.display = 'none';
                notification.remove();
            }, 3000);
        }
    </script>
    """,
    unsafe_allow_html=True
)

# Initialize session state
if 'expenses' not in st.session_state:
    st.session_state.expenses = []
if 'profit' not in st.session_state:
    st.session_state.profit = []
if 'menu' not in st.session_state:
    st.session_state.menu = "Home"

# User registration
if 'user_info' not in st.session_state:
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.title("🌱 Bhoomi - Farmer Registration")
    with st.form("user_form"):
        name = st.text_input("👤 Full Name")
        mobile = st.text_input("📞 Mobile Number")
        place = st.text_input("🏡 Place")
        submitted = st.form_submit_button("Submit 🚀")
    if submitted and name and mobile and place:
        st.session_state.user_info = {"name": name, "mobile": mobile, "place": place}
        st.session_state.menu = "Home"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.title(f"🌱 Bhoomi - Welcome {st.session_state.user_info['name']}")

    # Sidebar Navigation
    with st.sidebar:
        st.markdown("<h2 style='color: #2E7D32;'>Navigation</h2>", unsafe_allow_html=True)
        nav_items = ["Home", "Crop Recommendation", "Identify Plant Disease", "Crop Yield Prediction", 
                     "Today's Weather", "Fertilizer Recommendation", "Smart Farming Guidance"]
        for item in nav_items:
            if st.button(item, key=item):
                st.session_state.menu = item
                st.rerun()

    selected_menu = st.session_state.menu

    # Page Content
    if selected_menu == "Home":
        st.subheader("📊 Financial Overview")
        with st.form("finance_form"):
            finance_type = st.selectbox("📋 Type:", ["Expense", "Profit"])
            col1, col2 = st.columns(2)
            with col1:
                date = st.date_input(f"📅 {finance_type} Date", value=datetime.today())
            with col2:
                amount = st.number_input(f"💰 {finance_type} Amount", min_value=0.0, step=0.1)
            if finance_type == "Expense":
                purpose = st.text_input("📝 Expense For")
            submitted = st.form_submit_button("Add 🚀")
            if submitted:
                if finance_type == "Expense" and amount >= 0 and purpose:
                    st.session_state.expenses.append({"date": date.strftime('%Y-%m-%d'), "amount": amount, "purpose": purpose})
                    st.markdown(f'<script>showNotification("✅ Expense Added: <span>₹{amount}</span>");</script>', unsafe_allow_html=True)
                elif finance_type == "Profit" and amount >= 0:
                    st.session_state.profit.append({"date": date.strftime('%Y-%m-%d'), "amount": amount})
                    st.markdown(f'<script>showNotification("✅ Profit Added: <span>₹{amount}</span>");</script>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("💸 Expenses")
            if st.session_state.expenses:
                df_expenses = pd.DataFrame(st.session_state.expenses)
                st.table(df_expenses)
                st.markdown(f"**Total:** ₹{df_expenses['amount'].sum():.2f}")
        with col2:
            st.subheader("💰 Profits")
            if st.session_state.profit:
                df_profit = pd.DataFrame(st.session_state.profit)
                st.table(df_profit)
                st.markdown(f"**Total:** ₹{df_profit['amount'].sum():.2f}")

    elif selected_menu == "Crop Recommendation":
        st.subheader("🌾 Crop Recommendation")
        with st.form("crop_form"):
            col1, col2 = st.columns(2)
            with col1:
                nitrogen = st.number_input("🌿 Nitrogen (kg/ha)", min_value=0.0, step=0.1)
                phosphorus = st.number_input("🌱 Phosphorus (kg/ha)", min_value=0.0, step=0.1)
                potassium = st.number_input("🌿 Potassium (kg/ha)", min_value=0.0, step=0.1)
                temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
            with col2:
                humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
                ph = st.number_input("⚗️ pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
                rainfall = st.number_input("☔ Rainfall (mm)", min_value=0.0, step=0.1)
            submitted = st.form_submit_button("Predict 🌟")
        if submitted and crop_model:
            features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
            prediction = crop_model.predict(features)
            st.markdown(f'<script>showNotification("🌟 Recommended Crop: <span>{prediction[0]}</span>");</script>', unsafe_allow_html=True)

    elif selected_menu == "Identify Plant Disease":
        st.subheader("🦠 Plant Disease Identification")
        uploaded_file = st.file_uploader("📷 Upload Image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            disease = predict_disease(image)
            st.markdown(f'<script>showNotification("🌟 Detected Disease: <span>{disease}</span>");</script>', unsafe_allow_html=True)

    elif selected_menu == "Crop Yield Prediction":
        st.subheader("📊 Crop Yield Prediction")
        with st.form("yield_form"):
            col1, col2 = st.columns(2)
            with col1:
                country = st.selectbox("🌍 Country:", ["India", "Brazil", "USA"])
                rainfall = st.number_input("💧 Rainfall (mm/year)", min_value=0.0, step=0.1)
            with col2:
                crop = st.selectbox("🌾 Crop:", ["Maize", "Wheat", "Rice"])
                pesticide = st.number_input("🛡️ Pesticide (tonnes)", min_value=0.0, step=0.1)
            temperature = st.number_input("🌡️ Temperature (°C)", min_value=-50.0, max_value=50.0, step=0.1)
            submitted = st.form_submit_button("Predict 🚀")
        if submitted:
            st.markdown('<script>showNotification("🛠️ Yield Prediction: <span>5.0 tons (placeholder)</span>");</script>', unsafe_allow_html=True)

    elif selected_menu == "Today's Weather":
        st.subheader("🌤️ Weather Forecast")
        with st.form("weather_form"):
            col1, col2 = st.columns(2)
            with col1:
                zip_code = st.text_input("📍 ZIP Code")
            with col2:
                country_code = st.text_input("🌍 Country Code", value="IN")
            submitted = st.form_submit_button("Get Weather 🌞")
        if submitted:
            weather_data = get_weather(zip_code, country_code)
            if "main" in weather_data:
                temp = weather_data['main']['temp']
                st.markdown(f'<script>showNotification("🌡️ Temperature: <span>{temp}°C</span>");</script>', unsafe_allow_html=True)

    elif selected_menu == "Fertilizer Recommendation":
        st.subheader("🧪 Fertilizer Recommendation")
        with st.form("fertilizer_form"):
            col1, col2 = st.columns(2)
            with col1:
                temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
                humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
                moisture = st.number_input("💦 Moisture (%)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
                soil_type = st.selectbox("🌍 Soil Type", ["Sandy", "Loamy", "Black"])
            with col2:
                crop_type = st.selectbox("🌾 Crop Type", ["Maize", "Wheat", "Paddy"])
                nitrogen = st.number_input("🌿 Nitrogen (kg/ha)", min_value=0.0, step=0.1)
                potassium = st.number_input("🌿 Potassium (kg/ha)", min_value=0.0, step=0.1)
                phosphorous = st.number_input("🌱 Phosphorous (kg/ha)", min_value=0.0, step=0.1)
            submitted = st.form_submit_button("Recommend 🌟")
        if submitted and fertilizer_model:
            soil_encoded = label_encoder_soil.transform([soil_type])[0]
            crop_encoded = label_encoder_crop.transform([crop_type])[0]
            features = np.array([[temperature, humidity, moisture, soil_encoded, crop_encoded, nitrogen, potassium, phosphorous]])
            prediction = fertilizer_model.predict(features)
            st.markdown(f'<script>showNotification("🌟 Recommended Fertilizer: <span>{prediction[0]}</span>");</script>', unsafe_allow_html=True)

    elif selected_menu == "Smart Farming Guidance":
        st.subheader("📚 Smart Farming Guidance")
        with st.form("guidance_form"):
            col1, col2 = st.columns(2)
            with col1:
                crop = st.text_input("🌾 Crop Name")
            with col2:
                country = st.text_input("🌍 Country Name")
            submitted = st.form_submit_button("Get Guidance 🚀")
        if submitted:
            guidance = get_smart_farming_info(crop, country)
            st.markdown(f'<script>showNotification("📚 Guidance for <span>{crop}</span> retrieved");</script>', unsafe_allow_html=True)
            st.markdown(guidance, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
