import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Create a models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Load the dataset
data = pd.read_csv('datasets/Fertilizer Prediction.csv')

# Print the actual column names to confirm
print("Actual column names in the dataset:", data.columns.tolist())

# Encode categorical variables (Soil_Type and Crop_Type)
label_encoder_soil = LabelEncoder()
label_encoder_crop = LabelEncoder()
data['Soil_Type'] = label_encoder_soil.fit_transform(data['Soil_Type'])
data['Crop_Type'] = label_encoder_crop.fit_transform(data['Crop_Type'])

# Features and target (updated to match dataset column names)
X = data[['Temparature', 'Humidity', 'Moisture', 'Soil_Type', 'Crop_Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
y = data['Fertilizer']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model
joblib.dump(model, 'models/fertilizer_recommendation_model.pkl')
print("Model saved as 'models/fertilizer_recommendation_model.pkl'")

# Save the label encoders for later use in the app
joblib.dump(label_encoder_soil, 'models/label_encoder_soil.pkl')
joblib.dump(label_encoder_crop, 'models/label_encoder_crop.pkl')
print("Label encoders saved as 'models/label_encoder_soil.pkl' and 'models/label_encoder_crop.pkl'")
