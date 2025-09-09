from django.shortcuts import render, redirect, get_object_or_404
from .forms import PlantForm, RegisterForm, LoginForm
from .models import Plant, CareRecommendation, Profile
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm

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
            # Dummy recommendation for now
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
