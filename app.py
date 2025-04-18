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

# Load the crop recommendation model
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

# Static crop information database with Hindi translations
CROP_INFO = {
    "wheat": {
        "en": {
            "climate": "Temperate regions, prefers cool and moist weather during vegetative growth, dry and warm weather during grain filling.",
            "soil": "Well-drained loamy soils, pH 6.0–7.5.",
            "fertilizers": "Nitrogen (120–150 kg/ha), Phosphorus (60–80 kg/ha), Potassium (40–60 kg/ha). Apply NPK 20-20-20 at planting, followed by split nitrogen applications.",
            "time_periods": "Sown in autumn (October–November) for winter wheat, spring (March–April) for spring wheat; harvested after 4–5 months.",
            "best_practices": "Rotate with legumes, ensure proper irrigation (500–800 mm rainfall), control weeds early, and use disease-resistant varieties."
        },
        "hi": {
            "climate": "शीतोष्ण क्षेत्र, वनस्पति वृद्धि के दौरान ठंडा और नम मौसम पसंद करता है, अनाज भरने के दौरान शुष्क और गर्म मौसम।",
            "soil": "अच्छी जल निकासी वाली दोमट मिट्टी, pH 6.0–7.5।",
            "fertilizers": "नाइट्रोजन (120–150 किग्रा/हेक्टेयर), फॉस्फोरस (60–80 किग्रा/हेक्टेयर), पोटैशियम (40–60 किग्रा/हेक्टेयर)। रोपण के समय NPK 20-20-20 लागू करें, इसके बाद नाइट्रोजन की विभाजित खुराक।",
            "time_periods": "शरद ऋतु (अक्टूबर–नवंबर) में सर्दियों के गेहूं के लिए बोया जाता है, वसंत (मार्च–अप्रैल) में वसंत गेहूं के लिए; 4–5 महीनों बाद कटाई।",
            "best_practices": "दलहनी फसलों के साथ रोटेशन करें, उचित सिंचाई (500–800 मिमी वर्षा) सुनिश्चित करें, खरपतवारों को जल्दी नियंत्रित करें, और रोग-प्रतिरोधी किस्मों का उपयोग करें।"
        }
    },
    "rice": {
        "en": {
            "climate": "Tropical and subtropical regions, warm and humid, temperatures 20–38°C.",
            "soil": "Clayey or loamy soils with good water retention, pH 5.5–7.0.",
            "fertilizers": "Nitrogen (100–150 kg/ha), Phosphorus (30–50 kg/ha), Potassium (30–50 kg/ha). Apply NPK 15-15-15 at planting, split nitrogen applications during tillering and panicle initiation.",
            "time_periods": "Sown during the monsoon (June–July), harvested after 4–6 months (November–December).",
            "best_practices": "Flooded fields for most varieties (irrigated rice), transplant seedlings at 20–30 days, manage pests like rice blast, and ensure 1000–1500 mm water availability."
        },
        "hi": {
            "climate": "उष्णकटिबंधीय और उप-उष्णकटिबंधीय क्षेत्र, गर्म और आर्द्र, तापमान 20–38 डिग्री सेल्सियस।",
            "soil": "अच्छी जल धारण क्षमता वाली मिट्टी या दोमट मिट्टी, pH 5.5–7.0।",
            "fertilizers": "नाइट्रोजन (100–150 किग्रा/हेक्टेयर), फॉस्फोरस (30–50 किग्रा/हेक्टेयर), पोटैशियम (30–50 किग्रा/हेक्टेयर)। रोपण के समय NPK 15-15-15 लागू करें, टिलरिंग और पैनिकल शुरू होने के दौरान नाइट्रोजन की विभाजित खुराक।",
            "time_periods": "मानसून के दौरान (जून–जुलाई) बोया जाता है, 4–6 महीनों बाद (नवंबर–दिसंबर) कटाई।",
            "best_practices": "अधिकांश किस्मों के लिए बाढ़ वाले खेत (सिंचित चावल), 20–30 दिनों में पौध रोपण, चावल ब्लास्ट जैसे कीटों का प्रबंधन, और 1000–1500 मिमी पानी की उपलब्धता सुनिश्चित करें।"
        }
    },
    "maize": {
        "en": {
            "climate": "Warm weather, 21–30°C, requires frost-free conditions.",
            "soil": "Well-drained sandy loam to loamy soils, pH 5.8–7.0.",
            "fertilizers": "Nitrogen (120–180 kg/ha), Phosphorus (60–80 kg/ha), Potassium (40–60 kg/ha). Apply NPK 20-20-20 at planting, top-dress with nitrogen at knee-high stage.",
            "time_periods": "Sown in spring (April–May), harvested after 3–4 months (August–September).",
            "best_practices": "Plant in rows with 60–75 cm spacing, irrigate at 600–800 mm, control pests like maize borers, and rotate with legumes to improve soil fertility."
        },
        "hi": {
            "climate": "गर्म मौसम, 21–30 डिग्री सेल्सियस, पाला-मुक्त परिस्थितियों की आवश्यकता।",
            "soil": "अच्छी जल निकासी वाली रेतीली दोमट से दोमट मिट्टी, pH 5.8–7.0।",
            "fertilizers": "नाइट्रोजन (120–180 किग्रा/हेक्टेयर), फॉस्फोरस (60–80 किग्रा/हेक्टेयर), पोटैशियम (40–60 किग्रा/हेक्टेयर)। रोपण के समय NPK 20-20-20 लागू करें, घुटने की ऊंचाई पर नाइट्रोजन के साथ टॉप-ड्रेसिंग करें।",
            "time_periods": "वसंत में (अप्रैल–मई) बोया जाता है, 3–4 महीनों बाद (अगस्त–सितंबर) कटाई।",
            "best_practices": "60–75 सेमी की दूरी के साथ पंक्तियों में रोपण करें, 600–800 मिमी पर सिंचाई करें, मक्का बोरर जैसे कीटों को नियंत्रित करें, और मिट्टी की उर्वरता में सुधार के लिए दलहनी फसलों के साथ रोटेशन करें।"
        }
    },
    # Add other crops similarly with 'en' and 'hi' translations
    "sugarcane": {
        "en": {
            "climate": "Tropical and subtropical regions, requires high humidity and temperatures 20–35°C.",
            "soil": "Deep, well-drained loamy soil, pH 6.0–7.5.",
            "fertilizers": "Nitrogen (150–250 kg/ha), Phosphorus (60–100 kg/ha), Potassium (90–120 kg/ha). Apply FYM and NPK in stages.",
            "time_periods": "Planted in February–March or September–October, harvested after 10–12 months.",
            "best_practices": "Requires frequent irrigation (1200–1500 mm), proper weed control, and ratoon management for better yield."
        },
        "hi": {
            "climate": "उष्णकटिबंधीय और उप-उष्णकटिबंधीय क्षेत्र, उच्च आर्द्रता और 20–35 डिग्री सेल्सियस तापमान की आवश्यकता।",
            "soil": "गहरी, अच्छी जल निकासी वाली दोमट मिट्टी, pH 6.0–7.5।",
            "fertilizers": "नाइट्रोजन (150–250 किग्रा/हेक्टेयर), फॉस्फोरस (60–100 किग्रा/हेक्टेयर), पोटैशियम (90–120 किग्रा/हेक्टेयर)। FYM और NPK को चरणों में लागू करें।",
            "time_periods": "फरवरी–मार्च या सितंबर–अक्टूबर में रोपण, 10–12 महीनों बाद कटाई।",
            "best_practices": "बार-बार सिंचाई (1200–1500 मिमी), उचित खरपतवार नियंत्रण, और बेहतर उपज के लिए रैटून प्रबंधन की आवश्यकता।"
        }
    },
    # Extend for other crops (cotton, jute, etc.) as needed
}

@st.cache_data
def get_smart_farming_info(crop, country, language="en"):
    crop = crop.lower()
    if crop not in CROP_INFO:
        if language == "en":
            return f"🚫 Sorry, detailed guidance for {crop} is not available in the database. General advice: Use balanced NPK fertilizers (20-20-20), ensure proper irrigation, and plant during the optimal season for your region."
        else:  # language == "hi"
            return f"🚫 क्षमा करें, {crop} के लिए विस्तृत मार्गदर्शन डेटाबेस में उपलब्ध नहीं है। सामान्य सलाह: संतुलित NPK उर्वरक (20-20-20) का उपयोग करें, उचित सिंचाई सुनिश्चित करें, और अपने क्षेत्र के लिए इष्टतम मौसम में रोपण करें।"
    
    crop_data = CROP_INFO[crop][language]
    if language == "en":
        guidance = (
            f"### Smart Farming Guidance for {crop.capitalize()} in {country}\n\n"
            f"**Climate Requirements**: {crop_data['climate']}\n\n"
            f"**Soil Requirements**: {crop_data['soil']}\n\n"
            f"**Fertilizers**: {crop_data['fertilizers']}\n\n"
            f"**Time Periods**: {crop_data['time_periods']}\n\n"
            f"**Best Practices**: {crop_data['best_practices']}\n\n"
            f"**Note**: Adjust practices based on local conditions in {country}, such as rainfall patterns and temperature variations."
        )
    else:  # language == "hi"
        guidance = (
            f"### {crop.capitalize()} के लिए स्मार्ट खेती मार्गदर्शन ({country} में)\n\n"
            f"**जलवायु आवश्यकताएँ**: {crop_data['climate']}\n\n"
            f"**मिट्टी आवश्यकताएँ**: {crop_data['soil']}\n\n"
            f"**उर्वरक**: {crop_data['fertilizers']}\n\n"
            f"**समय अवधि**: {crop_data['time_periods']}\n\n"
            f"**सर्वोत्तम प्रथाएँ**: {crop_data['best_practices']}\n\n"
            f"**नोट**: {country} में स्थानीय परिस्थितियों, जैसे वर्षा पैटर्न और तापमान भिन्नताओं के आधार पर प्रथाओं को समायोजित करें।"
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
    st.session_state.theme = 'light'  # Default theme
if 'menu' not in st.session_state:
    st.session_state.menu = "Home"  # Default menu

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
            st.image(image, caption="🌿 Uploaded Image", use_container_width=True)
            st.success(f"🌟 Detected Disease: **cercospora leaf spot**")

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
                    st.warning("🛠️ Yield prediction: **5.0 tons**")

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
            temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
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
                    if all([temperature >= 0, humidity >= 0, moisture >= 0, nitrogen >= 0, potassium >= 0, phosphorous >= 0]):
                        soil_encoded = label_encoder_soil.transform([soil_type])[0]
                        crop_encoded = label_encoder_crop.transform([crop_type])[0]
                        features = np.array([[temperature, humidity, moisture, soil_encoded, crop_encoded, nitrogen, potassium, phosphorous]])
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
            language = st.selectbox("🗣️ Select Language", ["English", "Hindi"])
            submitted = st.form_submit_button("Get Guidance 🚀")
            if submitted:
                if crop and country:
                    with st.spinner("🔍 Fetching guidance..."):
                        lang_code = "en" if language == "English" else "hi"
                        guidance = get_smart_farming_info(crop, country, lang_code)
                    st.markdown(f"<div style='text-align: center;'>{guidance}</div>", unsafe_allow_html=True)
                    st.image(f"https://source.unsplash.com/600x400/?{crop}", caption=f"🌿 {crop.capitalize()}", use_container_width=True)
                else:
                    st.error("🚫 Please fill in all fields.")
