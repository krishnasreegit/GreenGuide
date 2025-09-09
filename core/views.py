from django.shortcuts import render, redirect, get_object_or_404
from .forms import PlantForm, RegisterForm, LoginForm
from .models import Plant, CareRecommendation, Profile
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
import joblib
import pandas as pd
import os
from django.conf import settings
from django.http import HttpResponse

# Load model once (global) - using absolute paths
try:
    model_path = os.path.join(settings.BASE_DIR, 'data', 'plant_knn_model.pkl')
    scaler_path = os.path.join(settings.BASE_DIR, 'data', 'scaler.pkl')
    encoders_path = os.path.join(settings.BASE_DIR, 'data', 'encoders.pkl')
    
    knn = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoders = joblib.load(encoders_path)
    MODEL_LOADED = True
except Exception as e:
    print(f"Error loading ML model: {e}")
    MODEL_LOADED = False
# Create your views here.

def home(request):
    return render(request, 'core/home.html')

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
        # Ensure email is saved
        user.email = form.cleaned_data.get('email', '')
        user.save()
        
        # Create profile with additional information
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


def plant_form(request):
    if request.method == 'POST':
        form = PlantForm(request.POST)
        if form.is_valid():
            plant = form.save()
            
            # Get ML recommendation if model is loaded
            if MODEL_LOADED:
                try:
                    # Convert watering_frequency to numeric for ML
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
                    
                    # Prepare data for ML model
                    plant_data = {
                        "growth": plant.growth,
                        "soil_type": plant.soil_type,
                        "sunlight": plant.sunlight,
                        "watering_numeric": convert_watering_to_numeric(plant.watering_frequency),
                        "fertilizer_type": plant.fertilizer_type
                    }
                    
                    # Encode categorical features
                    encoded_data = []
                    for col in ['growth', 'soil_type', 'sunlight', 'fertilizer_type']:
                        if plant_data[col] in encoders[col].classes_:
                            encoded_data.append(encoders[col].transform([plant_data[col]])[0])
                        else:
                            # Handle unknown categories by using the most common class
                            encoded_data.append(0)
                    encoded_data.append(plant_data['watering_numeric'])
                    
                    # Scale and predict
                    X_input = scaler.transform([encoded_data])
                    prediction = knn.predict(X_input)[0]
                    
                    # Create recommendation with ML prediction
                    rec = CareRecommendation.objects.create(
                        plant=plant,
                        watering_schedule=prediction,
                        sunlight_amount=f'{plant.sunlight}',
                        fertilizer_usage=f'{plant.fertilizer_type} fertilizer recommended',
                        seasonal_care=f'Growth rate: {plant.growth} - Adjust care seasonally'
                    )
                except Exception as e:
                    print(f"ML prediction error: {e}")
                    # Fallback to default recommendation
                    rec = CareRecommendation.objects.create(
                        plant=plant,
                        watering_schedule='Every 3 days',
                        sunlight_amount='6 hours/day',
                        fertilizer_usage='Once a month',
                        seasonal_care='Reduce watering in winter'
                    )
            else:
                # Default recommendation if model not loaded
                rec = CareRecommendation.objects.create(
                    plant=plant,
                    watering_schedule='Every 3 days',
                    sunlight_amount='6 hours/day',
                    fertilizer_usage='Once a month',
                    seasonal_care='Reduce watering in winter'
                )
            
            return redirect('care_chart', plant_id=plant.id)
    else:
        form = PlantForm()
    return render(request, 'core/plant_form.html', {'form': form})


def care_chart(request, plant_id):
    plant = get_object_or_404(Plant, id=plant_id)
    rec = CareRecommendation.objects.filter(plant=plant).last()
    return render(request, 'core/care_chart.html', {'plant': plant, 'rec': rec})


def chatbot(request):
    response = None
    if request.method == 'POST':
        question = request.POST.get('question')
        # Dummy response for now
        response = "This is a sample response from the AI assistant."
    return render(request, 'core/chatbot.html', {'response': response})


# Admin Dashboard - staff only
@user_passes_test(lambda u: u.is_staff)
def admin_dashboard(request):
    total_plants = Plant.objects.count()
    total_recommendations = CareRecommendation.objects.count()
    return render(request, 'core/admin_dashboard.html', {
        'total_plants': total_plants,
        'total_recommendations': total_recommendations,
    })

