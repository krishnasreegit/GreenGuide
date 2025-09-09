from django.shortcuts import render, redirect, get_object_or_404
from .forms import PlantForm, RegisterForm, LoginForm
from .models import Plant, CareRecommendation, Profile
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
import joblib
import pandas as pd
import numpy as np
import os
from django.conf import settings
from django.http import HttpResponse
from django.contrib import messages
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Load enhanced model artifacts once (global)
try:
    # Try to load the complete model artifacts first
    complete_model_path = os.path.join(settings.BASE_DIR, 'data', 'plant_model_complete.pkl')
    if os.path.exists(complete_model_path):
        model_artifacts = joblib.load(complete_model_path)
        ml_model = model_artifacts['model']
        scaler = model_artifacts['scaler']
        encoders = model_artifacts['encoders']
        feature_selector = model_artifacts.get('feature_selector')
        selected_features = model_artifacts.get('selected_features', [])
        target_mapping = model_artifacts.get('target_mapping', {})
        model_type = model_artifacts.get('model_type', 'Unknown')
        model_accuracy = model_artifacts.get('accuracy', 0.0)
        feature_names = model_artifacts.get('feature_names', [])
        
        print(f"Loaded enhanced model: {model_type} with accuracy: {model_accuracy:.4f}")
        MODEL_LOADED = True
        ENHANCED_MODEL = True
    else:
        # Fallback to individual files
        model_path = os.path.join(settings.BASE_DIR, 'data', 'plant_knn_model.pkl')
        scaler_path = os.path.join(settings.BASE_DIR, 'data', 'scaler.pkl')
        encoders_path = os.path.join(settings.BASE_DIR, 'data', 'encoders.pkl')
        
        ml_model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        encoders = joblib.load(encoders_path)
        feature_selector = None
        selected_features = []
        target_mapping = {}
        model_type = "KNN"
        model_accuracy = 0.0
        feature_names = []
        
        MODEL_LOADED = True
        ENHANCED_MODEL = False
        print("Loaded basic model artifacts")
        
except Exception as e:
    print(f"Error loading ML model: {e}")
    logger.error(f"ML model loading failed: {e}")
    MODEL_LOADED = False
    ENHANCED_MODEL = False

def home(request):
    """Enhanced home view with model information"""
    context = {
        'model_loaded': MODEL_LOADED,
        'model_type': model_type if MODEL_LOADED else 'None',
        'model_accuracy': f"{model_accuracy:.2%}" if MODEL_LOADED else 'N/A'
    }
    return render(request, 'core/home.html', context)

# Auth views
def login_view(request):
    if request.user.is_authenticated:
        return redirect('home')
    form = LoginForm(request, data=request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = form.get_user()
        login(request, user)
        return redirect('home')
    return render(request, 'core/login.html', {'form': form})

def register_view(request):
    if request.user.is_authenticated:
        return redirect('home')
    
    form = RegisterForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = form.save(commit=False)
        user.email = form.cleaned_data.get('email', '')
        user.save()
        
        Profile.objects.create(
            user=user,
            location=form.cleaned_data.get('location', ''),
            gender=form.cleaned_data.get('gender', ''),
            age=form.cleaned_data.get('age') if form.cleaned_data.get('age') not in [None, ''] else None,
        )
        
        login(request, user)
        return redirect('home')
    
    return render(request, 'core/register.html', {'form': form})

@login_required
def logout_view(request):
    logout(request)
    return redirect('home')

def create_care_complexity_score(plant):
    """Enhanced care complexity calculation"""
    complexity = 0
    
    # Growth rate contribution
    growth = plant.growth.lower()
    if growth == 'fast':
        complexity += 3
    elif growth == 'moderate':
        complexity += 2
    else:  # slow
        complexity += 1
    
    # Soil type contribution
    soil = plant.soil_type.lower()
    if any(term in soil for term in ['sandy', 'well-drained']):
        complexity += 1
    elif any(term in soil for term in ['loamy', 'moist']):
        complexity += 2
    else:  # acidic or other
        complexity += 3
    
    # Sunlight contribution
    sunlight = plant.sunlight.lower()
    if 'full sunlight' in sunlight:
        complexity += 1
    elif 'partial sunlight' in sunlight:
        complexity += 2
    else:  # indirect
        complexity += 3
        
    return complexity

def convert_watering_to_numeric_enhanced(text):
    """Enhanced watering frequency conversion"""
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

def group_watering_schedules(schedule):
    """Group similar watering schedules for better prediction"""
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

def get_enhanced_ml_prediction(plant):
    """Enhanced ML prediction using the optimized model"""
    try:
        if not MODEL_LOADED:
            return None
            
        # Clean and prepare plant data
        plant_data = {
            'growth': plant.growth.lower().strip(),
            'soil_type': plant.soil_type.lower().strip(),
            'sunlight': plant.sunlight.lower().strip(),
            'fertilizer_type': plant.fertilizer_type.lower().strip(),
            'watering_numeric': convert_watering_to_numeric_enhanced(plant.watering_frequency),
            'care_complexity': create_care_complexity_score(plant)
        }
        
        if ENHANCED_MODEL:
            # Use the enhanced model with one-hot encoding
            encoded_features = []
            categorical_cols = ['growth', 'soil_type', 'sunlight', 'fertilizer_type']
            
            # One-hot encode categorical features
            for col in categorical_cols:
                if col in encoders:
                    try:
                        encoded = encoders[col].transform([[plant_data[col]]])
                        feature_names = [f"{col}_{cat}" for cat in encoders[col].categories_[0]]
                        encoded_df = pd.DataFrame(encoded, columns=feature_names)
                        encoded_features.append(encoded_df)
                    except Exception as e:
                        logger.warning(f"Error encoding {col}: {e}")
                        # Create zero vector for unknown categories
                        n_categories = len(encoders[col].categories_[0])
                        zero_encoded = np.zeros((1, n_categories))
                        feature_names = [f"{col}_{cat}" for cat in encoders[col].categories_[0]]
                        encoded_df = pd.DataFrame(zero_encoded, columns=feature_names)
                        encoded_features.append(encoded_df)
            
            # Combine all features
            if encoded_features:
                encoded_combined = pd.concat(encoded_features, axis=1)
                numerical_features = pd.DataFrame({
                    'watering_numeric': [plant_data['watering_numeric']],
                    'care_complexity': [plant_data['care_complexity']]
                })
                X_input = pd.concat([encoded_combined, numerical_features], axis=1)
                
                # Apply feature selection if available
                if feature_selector is not None:
                    X_input = feature_selector.transform(X_input.abs())
                
                # Scale features
                if model_type == 'Random Forest':
                    # Random Forest doesn't need scaling
                    prediction = ml_model.predict(X_input)[0]
                else:
                    X_input_scaled = scaler.transform(X_input)
                    prediction = ml_model.predict(X_input_scaled)[0]
                
                # Get confidence score if available
                try:
                    if hasattr(ml_model, 'predict_proba'):
                        if model_type == 'Random Forest':
                            probabilities = ml_model.predict_proba(X_input)[0]
                        else:
                            probabilities = ml_model.predict_proba(X_input_scaled)[0]
                        confidence = np.max(probabilities)
                    else:
                        confidence = 0.8  # Default confidence
                except:
                    confidence = 0.8
                
                return {
                    'prediction': prediction,
                    'confidence': confidence,
                    'model_type': model_type
                }
                
        else:
            # Use basic model (fallback)
            encoded_data = []
            categorical_cols = ['growth', 'soil_type', 'sunlight', 'fertilizer_type']
            
            for col in categorical_cols:
                if col in encoders:
                    try:
                        if plant_data[col] in encoders[col].classes_:
                            encoded_data.append(encoders[col].transform([plant_data[col]])[0])
                        else:
                            encoded_data.append(0)  # Unknown category
                    except:
                        encoded_data.append(0)
                else:
                    encoded_data.append(0)
            
            encoded_data.append(plant_data['watering_numeric'])
            
            # Scale and predict
            X_input = scaler.transform([encoded_data])
            prediction = ml_model.predict(X_input)[0]
            
            return {
                'prediction': prediction,
                'confidence': 0.7,  # Default confidence for basic model
                'model_type': 'KNN'
            }
            
    except Exception as e:
        logger.error(f"ML prediction error: {e}")
        return None

def create_intelligent_recommendation(plant, ml_result=None):
    """Create intelligent care recommendations based on plant characteristics and ML prediction"""
    
    # Base recommendations
    recommendations = {
        'watering_schedule': 'Water every 3-5 days',
        'sunlight_amount': '6 hours/day',
        'fertilizer_usage': 'Monthly fertilization',
        'seasonal_care': 'Adjust care seasonally'
    }
    
    if ml_result:
        # Use ML prediction
        predicted_schedule = ml_result['prediction']
        confidence = ml_result['confidence']
        model_used = ml_result['model_type']
        
        # Convert grouped prediction back to readable schedule
        schedule_mapping = {
            'daily_watering': 'Water daily',
            'weekly_watering': 'Water weekly',
            'keep_moist': 'Keep soil consistently moist',
            'dry_between_watering': 'Water when soil is dry',
            'regular_watering': 'Water regularly (every 3-5 days)',
            'other_schedule': 'Follow specific watering needs'
        }
        
        recommendations['watering_schedule'] = schedule_mapping.get(
            predicted_schedule, predicted_schedule
        )
        
        # Add confidence and model info
        recommendations['ml_confidence'] = f"{confidence:.1%}"
        recommendations['model_used'] = model_used
        
    # Enhance with plant-specific logic
    growth = plant.growth.lower()
    soil = plant.soil_type.lower()
    sunlight = plant.sunlight.lower()
    fertilizer = plant.fertilizer_type.lower()
    
    # Sunlight recommendations
    if 'full sunlight' in sunlight:
        recommendations['sunlight_amount'] = '6-8 hours direct sunlight daily'
    elif 'partial sunlight' in sunlight:
        recommendations['sunlight_amount'] = '4-6 hours sunlight daily'
    else:
        recommendations['sunlight_amount'] = 'Bright indirect light'
    
    # Fertilizer recommendations
    if fertilizer == 'organic':
        recommendations['fertilizer_usage'] = 'Organic fertilizer every 4-6 weeks'
    elif fertilizer == 'balanced':
        recommendations['fertilizer_usage'] = 'Balanced fertilizer monthly'
    elif fertilizer == 'low-nitrogen':
        recommendations['fertilizer_usage'] = 'Low-nitrogen fertilizer every 6-8 weeks'
    elif fertilizer == 'no' or fertilizer == 'none':
        recommendations['fertilizer_usage'] = 'No fertilizer needed'
    else:
        recommendations['fertilizer_usage'] = f'{fertilizer.title()} fertilizer as needed'
    
    # Seasonal care based on growth rate
    if growth == 'fast':
        recommendations['seasonal_care'] = 'Monitor frequently; increase feeding in growing season'
    elif growth == 'slow':
        recommendations['seasonal_care'] = 'Reduce watering in winter; minimal feeding needed'
    else:
        recommendations['seasonal_care'] = 'Moderate care adjustment with seasons'
    
    return recommendations

def plant_form(request):
    """Enhanced plant form with intelligent ML predictions"""
    if request.method == 'POST':
        form = PlantForm(request.POST)
        if form.is_valid():
            plant = form.save()
            
            try:
                # Get ML recommendation if model is loaded
                ml_result = None
                if MODEL_LOADED:
                    ml_result = get_enhanced_ml_prediction(plant)
                
                # Create intelligent recommendations
                recommendations = create_intelligent_recommendation(plant, ml_result)
                
                # Create recommendation object
                rec = CareRecommendation.objects.create(
                    plant=plant,
                    watering_schedule=recommendations['watering_schedule'],
                    sunlight_amount=recommendations['sunlight_amount'],
                    fertilizer_usage=recommendations['fertilizer_usage'],
                    seasonal_care=recommendations['seasonal_care']
                )
                
                # Add success message with model info
                if ml_result:
                    messages.success(
                        request, 
                        f"Care recommendations generated using {ml_result['model_type']} "
                        f"model with {ml_result['confidence']:.1%} confidence."
                    )
                else:
                    messages.info(
                        request,
                        "Care recommendations generated using rule-based system."
                    )
                
                return redirect('care_chart', plant_id=plant.id)
                
            except Exception as e:
                logger.error(f"Error creating recommendations: {e}")
                # Fallback to basic recommendation
                rec = CareRecommendation.objects.create(
                    plant=plant,
                    watering_schedule='Every 3-5 days',
                    sunlight_amount='6 hours/day',
                    fertilizer_usage='Monthly fertilization',
                    seasonal_care='Adjust care seasonally'
                )
                messages.warning(
                    request,
                    "Basic care recommendations provided due to system limitations."
                )
                return redirect('care_chart', plant_id=plant.id)
    else:
        form = PlantForm()
    
    context = {
        'form': form,
        'model_available': MODEL_LOADED,
        'model_type': model_type if MODEL_LOADED else 'None'
    }
    return render(request, 'core/plant_form.html', context)

def care_chart(request, plant_id):
    """Enhanced care chart with detailed recommendations"""
    plant = get_object_or_404(Plant, id=plant_id)
    rec = CareRecommendation.objects.filter(plant=plant).last()
    
    context = {
        'plant': plant,
        'rec': rec,
        'model_info': {
            'available': MODEL_LOADED,
            'type': model_type if MODEL_LOADED else 'None',
            'accuracy': f"{model_accuracy:.1%}" if MODEL_LOADED else 'N/A'
        }
    }
    return render(request, 'core/care_chart.html', context)

def chatbot(request):
    """Enhanced chatbot with plant-specific knowledge"""
    response = None
    if request.method == 'POST':
        question = request.POST.get('question', '').lower()
        
        # Simple rule-based responses for plant care
        if any(word in question for word in ['water', 'watering']):
            response = "Watering frequency depends on your plant type, soil, and environment. Most plants need water when the top inch of soil feels dry. Overwatering is more dangerous than underwatering for most plants."
        
        elif any(word in question for word in ['sunlight', 'light', 'sun']):
            response = "Light requirements vary by plant. 'Full sunlight' means 6+ hours of direct sun, 'partial sunlight' means 3-6 hours, and 'indirect sunlight' means bright light without direct sun exposure."
        
        elif any(word in question for word in ['fertilizer', 'fertilize', 'feed']):
            response = "Fertilization depends on your plant's growth rate and type. Fast-growing plants need more frequent feeding. Organic fertilizers release nutrients slowly, while balanced fertilizers provide equal NPK ratios."
        
        elif any(word in question for word in ['soil', 'potting']):
            response = "Soil type affects drainage and nutrients. Sandy soil drains quickly, loamy soil holds moderate moisture, and well-drained soil prevents water logging. Choose based on your plant's needs."
        
        elif any(word in question for word in ['growth', 'growing', 'slow', 'fast']):
            response = "Plant growth rates affect care frequency. Fast-growing plants need more water, nutrients, and attention. Slow-growing plants are more tolerant but need patience."
        
        else:
            response = "I can help with plant care questions about watering, sunlight, fertilization, soil types, and growth patterns. Feel free to ask specific questions about plant care!"
    
    return render(request, 'core/chatbot.html', {'response': response})

# Admin Dashboard - staff only
@user_passes_test(lambda u: u.is_staff)
def admin_dashboard(request):
    """Enhanced admin dashboard with model statistics"""
    total_plants = Plant.objects.count()
    total_recommendations = CareRecommendation.objects.count()
    
    # Get plant statistics
    plant_stats = {}
    if total_plants > 0:
        # Growth rate distribution
        from django.db.models import Count
        growth_stats = Plant.objects.values('growth').annotate(count=Count('growth'))
        plant_stats['growth'] = {item['growth']: item['count'] for item in growth_stats}
        
        # Sunlight distribution
        sunlight_stats = Plant.objects.values('sunlight').annotate(count=Count('sunlight'))
        plant_stats['sunlight'] = {item['sunlight']: item['count'] for item in sunlight_stats}
        
        # Soil type distribution
        soil_stats = Plant.objects.values('soil_type').annotate(count=Count('soil_type'))
        plant_stats['soil'] = {item['soil_type']: item['count'] for item in soil_stats}
    
    context = {
        'total_plants': total_plants,
        'total_recommendations': total_recommendations,
        'plant_stats': plant_stats,
        'model_info': {
            'loaded': MODEL_LOADED,
            'type': model_type if MODEL_LOADED else 'None',
            'enhanced': ENHANCED_MODEL,
            'accuracy': f"{model_accuracy:.2%}" if MODEL_LOADED else 'N/A'
        }
    }
    return render(request, 'core/admin_dashboard.html', context)

# Additional utility views for model management
@user_passes_test(lambda u: u.is_staff)
def model_status(request):
    """View to check model status and performance"""
    if not MODEL_LOADED:
        return HttpResponse("Model not loaded", status=503)
    
    status_info = {
        'model_loaded': MODEL_LOADED,
        'enhanced_model': ENHANCED_MODEL,
        'model_type': model_type,
        'accuracy': model_accuracy,
        'features_count': len(selected_features) if ENHANCED_MODEL else 'Unknown'
    }
    
    return render(request, 'core/model_status.html', {'status': status_info})

@login_required
def my_plants(request):
    """View user's plants and recommendations"""
    # This assumes you have a user field in Plant model
    # If not, you'll need to modify the Plant model
    try:
        user_plants = Plant.objects.filter(user=request.user) if hasattr(Plant, 'user') else Plant.objects.all()[:10]
    except:
        user_plants = Plant.objects.all()[:10]  # Fallback
    
    plants_with_recommendations = []
    for plant in user_plants:
        recommendation = CareRecommendation.objects.filter(plant=plant).last()
        plants_with_recommendations.append({
            'plant': plant,
            'recommendation': recommendation
        })
    
    context = {
        'plants_data': plants_with_recommendations,
        'total_plants': len(plants_with_recommendations)
    }
    return render(request, 'core/my_plants.html', context)

# Additional helper views
def plant_search(request):
    """Search plants by name or characteristics"""
    query = request.GET.get('q', '')
    plants = []
    
    if query:
        plants = Plant.objects.filter(
            name__icontains=query
        ) | Plant.objects.filter(
            growth__icontains=query
        ) | Plant.objects.filter(
            soil_type__icontains=query
        )
    
    return render(request, 'core/plant_search.html', {
        'plants': plants,
        'query': query
    })

def plant_detail(request, plant_id):
    """Detailed view of a specific plant"""
    plant = get_object_or_404(Plant, id=plant_id)
    recommendations = CareRecommendation.objects.filter(plant=plant)
    
    # Get ML prediction for this plant if model is available
    ml_result = None
    if MODEL_LOADED:
        ml_result = get_enhanced_ml_prediction(plant)
    
    context = {
        'plant': plant,
        'recommendations': recommendations,
        'ml_result': ml_result
    }
    return render(request, 'core/plant_detail.html', context)

# API-like views for mobile or external access
from django.http import JsonResponse

def api_plant_recommendation(request, plant_id):
    """API endpoint to get plant recommendations as JSON"""
    try:
        plant = get_object_or_404(Plant, id=plant_id)
        
        # Get ML prediction
        ml_result = None
        if MODEL_LOADED:
            ml_result = get_enhanced_ml_prediction(plant)
        
        # Create recommendations
        recommendations = create_intelligent_recommendation(plant, ml_result)
        
        # Get latest saved recommendation
        saved_rec = CareRecommendation.objects.filter(plant=plant).last()
        
        response_data = {
            'plant': {
                'id': plant.id,
                'name': plant.name,
                'growth': plant.growth,
                'soil_type': plant.soil_type,
                'sunlight': plant.sunlight,
                'fertilizer_type': plant.fertilizer_type
            },
            'recommendations': recommendations,
            'saved_recommendation': {
                'watering_schedule': saved_rec.watering_schedule if saved_rec else None,
                'sunlight_amount': saved_rec.sunlight_amount if saved_rec else None,
                'fertilizer_usage': saved_rec.fertilizer_usage if saved_rec else None,
                'seasonal_care': saved_rec.seasonal_care if saved_rec else None,
            } if saved_rec else None,
            'ml_prediction': ml_result,
            'model_info': {
                'available': MODEL_LOADED,
                'type': model_type if MODEL_LOADED else 'None',
                'enhanced': ENHANCED_MODEL
            }
        }
        
        return JsonResponse(response_data)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def api_model_status(request):
    """API endpoint to check model status"""
    return JsonResponse({
        'model_loaded': MODEL_LOADED,
        'enhanced_model': ENHANCED_MODEL,
        'model_type': model_type if MODEL_LOADED else 'None',
        'accuracy': model_accuracy if MODEL_LOADED else 0,
        'features_count': len(selected_features) if ENHANCED_MODEL else 0
    })