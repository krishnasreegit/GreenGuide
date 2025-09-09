from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Plant(models.Model):
    GROWTH_CHOICES = [
        ('slow', 'Slow'),
        ('moderate', 'Moderate'),
        ('fast', 'Fast'),
    ]
    
    FERTILIZER_CHOICES = [
        ('balanced', 'Balanced'),
        ('organic', 'Organic'),
        ('low-nitrogen', 'Low-nitrogen'),
        ('acidic', 'Acidic'),
        ('no', 'No fertilizer'),
    ]
    
    name = models.CharField(max_length=100)
    growth = models.CharField(max_length=20, choices=GROWTH_CHOICES, default='moderate')
    soil_type = models.CharField(max_length=100)
    sunlight = models.CharField(max_length=50)
    watering_frequency = models.CharField(max_length=100)  # Changed to text-based
    fertilizer_type = models.CharField(max_length=20, choices=FERTILIZER_CHOICES, default='balanced')

    def __str__(self):
        return self.name

class CareRecommendation(models.Model):
    plant = models.ForeignKey(Plant, on_delete=models.CASCADE)
    watering_schedule = models.CharField(max_length=100)
    sunlight_amount = models.CharField(max_length=100)
    fertilizer_usage = models.CharField(max_length=100)
    seasonal_care = models.CharField(max_length=200)

    def __str__(self):
        return f"Recommendations for {self.plant.name}"

# New user profile to store extra signup fields
class Profile(models.Model):
    GENDER_CHOICES = [
        ('male', 'Male'),
        ('female', 'Female'),
        ('other', 'Other'),
        ('prefer_not', 'Prefer not to say'),
    ]
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    location = models.CharField(max_length=120, blank=True)
    gender = models.CharField(max_length=20, choices=GENDER_CHOICES, blank=True)
    age = models.PositiveIntegerField(null=True, blank=True)

    def __str__(self):
        return f"Profile of {self.user.username}"
