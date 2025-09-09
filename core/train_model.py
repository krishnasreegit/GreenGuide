import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import os

# Path to dataset
csv_path = r"C:\sem_3_project\data\plants.csv"

# 1. Load dataset with proper encoding
df = pd.read_csv(csv_path, encoding="latin1")

# 2. Rename columns to match your model pipeline
df.rename(columns={
    "Plant Name": "name",
    "Growth": "growth",
    "Soil": "soil_type",
    "Sunlight": "sunlight",
    "Watering": "watering_frequency",
    "Fertilization Type": "fertilizer_type"
}, inplace=True)

# 3. Convert watering_frequency to numeric for ML (approximation)
def convert_watering_to_numeric(text):
    text = str(text).lower()
    if "daily" in text:
        return 1
    elif "twice" in text or "two" in text:
        return 3
    elif "weekly" in text:
        return 7
    elif "when soil is dry" in text or "let soil dry" in text:
        return 10
    elif "moist" in text:
        return 2
    elif "regular" in text:
        return 5
    else:
        return 5  # default mid value

# Create numeric version for ML training
df["watering_numeric"] = df["watering_frequency"].apply(convert_watering_to_numeric)

# 4. For target, we'll predict the watering_frequency text
# This will help provide more accurate recommendations
df["recommended_schedule"] = df["watering_frequency"]

# 5. Encode categorical features
label_cols = ['growth', 'soil_type', 'sunlight', 'fertilizer_type']
encoders = {}
for col in label_cols:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])

# 6. Features & Target
X = df[['growth', 'soil_type', 'sunlight', 'watering_numeric', 'fertilizer_type']]
y = df['recommended_schedule']

# 7. Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8. Train KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)

# 9. Save model & encoders
joblib.dump(knn, r"C:\sem_3_project\data\plant_knn_model.pkl")
joblib.dump(scaler, r"C:\sem_3_project\data\scaler.pkl")
joblib.dump(encoders, r"C:\sem_3_project\data\encoders.pkl")

print("✅ Model trained successfully on new plants.csv dataset and saved!")
print(f"Dataset shape: {df.shape}")
print(f"Features used: {list(X.columns)}")
print(f"Target classes: {len(y.unique())}")
