from django import forms
from .models import Plant
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User

class PlantForm(forms.ModelForm):
    class Meta:
        model = Plant
        fields = ['name', 'species', 'soil_type', 'sunlight', 'region', 'watering_frequency']

class RegisterForm(UserCreationForm):
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your email'
        })
    )
    location = forms.CharField(
        required=False, 
        max_length=120,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your location'
        })
    )
    gender = forms.ChoiceField(
        required=False, 
        choices=[
            ('', 'Select gender'),
            ('male', 'Male'),
            ('female', 'Female'),
            ('other', 'Other'),
            ('prefer_not', 'Prefer not to say'),
        ],
        widget=forms.Select(attrs={
            'class': 'form-control'
        })
    )
    age = forms.IntegerField(
        required=False, 
        min_value=0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your age'
        })
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add Bootstrap classes to default fields
        self.fields['username'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Choose a username'
        })
        self.fields['password1'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Create a password'
        })
        self.fields['password2'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Confirm your password'
        })

    class Meta:
        model = User
        # IMPORTANT: Only include fields that exist on User model
        fields = ['username', 'email', 'password1', 'password2']


class LoginForm(AuthenticationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add Bootstrap classes to form fields
        self.fields['username'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Enter your username'
        })
        self.fields['password'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Enter your password'
        }) 