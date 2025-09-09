import pandas as pd
import joblib
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif  # Changed from chi2 to f_classif
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Path to dataset
csv_path = r"C:\sem_3_project\data\plants.csv"

# 1. Load dataset with proper encoding and handling
print("Loading and preprocessing dataset...")
df = pd.read_csv(csv_path, encoding="latin1")

# Remove duplicates and clean data
df = df.drop_duplicates()
df = df.dropna()

# 2. Rename columns
df.rename(columns={
    "Plant Name": "name",
    "Growth": "growth", 
    "Soil": "soil_type",
    "Sunlight": "sunlight",
    "Watering": "watering_frequency",
    "Fertilization Type": "fertilizer_type"
}, inplace=True)

# 3. Enhanced watering frequency conversion with more nuanced mapping
def convert_watering_to_numeric(text):
    text = str(text).lower().strip()
    
    # More specific mappings for better accuracy
    if "daily" in text or "every day" in text:
        return 1
    elif "twice" in text or "two times" in text or "2 times" in text:
        return 3
    elif "weekly" in text or "once a week" in text:
        return 7
    elif "when soil is dry" in text or "let soil dry" in text or "dry between watering" in text:
        return 10
    elif "keep soil moist" in text or "consistently moist" in text or "evenly moist" in text:
        return 2
    elif "slightly moist" in text:
        return 4
    elif "regular" in text or "regular watering" in text:
        return 5
    elif "topsoil is dry" in text or "when topsoil" in text:
        return 6
    elif "feels dry" in text:
        return 8
    else:
        return 5  # default

# Apply improved conversion
df["watering_numeric"] = df["watering_frequency"].apply(convert_watering_to_numeric)

# 4. Create more sophisticated features
def create_care_complexity_score(row):
    """Create a composite feature representing care complexity"""
    complexity = 0
    
    # Growth rate contribution
    if row['growth'] == 'fast':
        complexity += 3
    elif row['growth'] == 'moderate':
        complexity += 2
    else:  # slow
        complexity += 1
    
    # Soil type contribution
    if row['soil_type'] in ['sandy', 'well-drained']:
        complexity += 1
    elif row['soil_type'] in ['loamy', 'moist']:
        complexity += 2
    else:  # acidic or other
        complexity += 3
    
    # Sunlight contribution
    if row['sunlight'] == 'full sunlight':
        complexity += 1
    elif row['sunlight'] == 'partial sunlight':
        complexity += 2
    else:  # indirect
        complexity += 3
        
    return complexity

df['care_complexity'] = df.apply(create_care_complexity_score, axis=1)

# 5. Advanced categorical encoding using One-Hot for better feature representation
categorical_cols = ['growth', 'soil_type', 'sunlight', 'fertilizer_type']
encoders = {}
encoded_features = []

# Use OneHot encoding for better performance
for col in categorical_cols:
    # Clean the column values
    df[col] = df[col].astype(str).str.strip().str.lower()
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[[col]])
    
    # Create feature names
    feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
    encoded_features.append(encoded_df)
    
    encoders[col] = encoder

# Combine all encoded features
encoded_combined = pd.concat(encoded_features, axis=1)

# 6. Create final feature matrix
numerical_features = df[['watering_numeric', 'care_complexity']]
X = pd.concat([encoded_combined, numerical_features], axis=1)
y = df['watering_frequency']

print(f"Feature matrix shape: {X.shape}")
print(f"Number of unique target classes: {len(y.unique())}")

# 7. Handle class imbalance by grouping similar watering schedules
def group_watering_schedules(schedule):
    schedule = str(schedule).lower()
    
    if any(term in schedule for term in ['daily', 'every day']):
        return 'daily_watering'
    elif any(term in schedule for term in ['weekly', 'week']):
        return 'weekly_watering'  
    elif any(term in schedule for term in ['moist', 'consistently', 'evenly']):
        return 'keep_moist'
    elif any(term in schedule for term in ['dry', 'let soil dry', 'when soil is dry']):
        return 'dry_between_watering'
    elif any(term in schedule for term in ['regular', 'regular watering']):
        return 'regular_watering'
    else:
        return 'other_schedule'

y_grouped = y.apply(group_watering_schedules)

# 8. CRITICAL FIX: Encode target labels to numeric values
print("Encoding target labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_grouped)
target_classes = label_encoder.classes_
print(f"Target classes: {target_classes}")

# 9. Feature selection using f_classif (better for mixed data types)
print("Performing feature selection...")
selector = SelectKBest(score_func=f_classif, k=min(18, X.shape[1]))  # Reduced k to match your successful run
X_selected = selector.fit_transform(X, y_encoded)  # Use encoded labels
selected_feature_names = X.columns[selector.get_support()].tolist()

print(f"Selected {X_selected.shape[1]} most important features")

# 10. Advanced train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_encoded,  # Use encoded labels
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded  # Use encoded labels
)

# 11. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 12. Comprehensive hyperparameter tuning
print("Finding optimal hyperparameters...")

# KNN hyperparameter grid (simplified to reduce computation)
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Random Forest parameters
rf_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

# Use StratifiedKFold for better cross-validation
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid search for KNN
knn_grid_search = GridSearchCV(
    KNeighborsClassifier(),
    knn_param_grid,
    cv=cv_strategy,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

knn_grid_search.fit(X_train_scaled, y_train)
best_knn = knn_grid_search.best_estimator_

print(f"Best KNN parameters: {knn_grid_search.best_params_}")
print(f"Best KNN CV score: {knn_grid_search.best_score_:.4f}")

# Grid search for Random Forest
rf_grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=cv_strategy,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

rf_grid_search.fit(X_train, y_train)  # RF can handle unscaled data
best_rf = rf_grid_search.best_estimator_

print(f"Best RF parameters: {rf_grid_search.best_params_}")
print(f"Best RF CV score: {rf_grid_search.best_score_:.4f}")

# 13. Create an ensemble model
print("Training ensemble model...")

# For ensemble, we need to create custom estimators that handle the data preprocessing
from sklearn.pipeline import Pipeline

# Create pipelines for consistent preprocessing
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', best_knn)
])

rf_pipeline = Pipeline([
    ('rf', best_rf)
])

voting_clf = VotingClassifier(
    estimators=[
        ('knn', knn_pipeline),
        ('rf', rf_pipeline)
    ],
    voting='soft'
)

voting_clf.fit(X_train, y_train)

# 14. Evaluate models
models = {
    'KNN': (best_knn, X_train_scaled, X_test_scaled),
    'Random Forest': (best_rf, X_train, X_test),
    'Ensemble': (voting_clf, X_train, X_test)
}

best_model = None
best_accuracy = 0
best_model_name = ""

print("\nModel Evaluation Results:")
print("=" * 50)

for name, (model, X_tr, X_te) in models.items():
    # Cross-validation score
    cv_scores = cross_val_score(model, X_tr, y_train, cv=cv_strategy, scoring='accuracy')
    
    # Test accuracy
    y_pred = model.predict(X_te)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{name}:")
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model = model
        best_model_name = name

print(f"\nBest performing model: {best_model_name} with accuracy: {best_accuracy:.4f}")

# 15. Final model training on full dataset
print("\nTraining final model on complete dataset...")
X_full_scaled = scaler.fit_transform(X_selected)

if best_model_name == 'Random Forest':
    final_model = best_rf
    final_model.fit(X_selected, y_encoded)
elif best_model_name == 'KNN':
    final_model = best_knn
    final_model.fit(X_full_scaled, y_encoded)
else:  # Ensemble
    final_model = voting_clf
    final_model.fit(X_selected, y_encoded)

# 16. Save all artifacts with enhanced metadata
model_artifacts = {
    'model': final_model,
    'scaler': scaler,
    'encoders': encoders,
    'label_encoder': label_encoder,  # IMPORTANT: Save label encoder
    'feature_selector': selector,
    'selected_features': selected_feature_names,
    'target_classes': target_classes,
    'model_type': best_model_name,
    'accuracy': best_accuracy,
    'feature_names': list(X.columns)
}

# Create directory if it doesn't exist
os.makedirs(r"C:\sem_3_project\data", exist_ok=True)

joblib.dump(model_artifacts, r"C:\sem_3_project\data\plant_model_complete.pkl")
joblib.dump(final_model, r"C:\sem_3_project\data\plant_knn_model.pkl")
joblib.dump(scaler, r"C:\sem_3_project\data\scaler.pkl")
joblib.dump(encoders, r"C:\sem_3_project\data\encoders.pkl")
joblib.dump(label_encoder, r"C:\sem_3_project\data\label_encoder.pkl")

print(f"\nModel training completed successfully!")
print(f"Dataset shape: {df.shape}")
print(f"Final features used: {X_selected.shape[1]}")
print(f"Target classes: {len(target_classes)}")
print(f"Final Model: {best_model_name}")
print(f"Final Accuracy: {best_accuracy:.4f}")
print(f"All artifacts saved to data directory.")

# Print feature importance for Random Forest if it was selected
if best_model_name == 'Random Forest' and hasattr(best_rf, 'feature_importances_'):
    print(f"\nTop 10 Most Important Features:")
    feature_importance = list(zip(selected_feature_names, best_rf.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    for feature, importance in feature_importance[:10]:
        print(f"  {feature}: {importance:.4f}")

# Print classification report for detailed evaluation
y_pred_final = final_model.predict(X_test if best_model_name != 'KNN' else X_test_scaled)
print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_final, 
                          target_names=label_encoder.inverse_transform(range(len(target_classes)))))