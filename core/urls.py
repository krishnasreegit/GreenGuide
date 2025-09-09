from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.home, name='home'),
    path('plant/', views.plant_form, name='plant_form'),
    path('care/<int:plant_id>/', views.care_chart, name='care_chart'),
    path('chatbot/', views.chatbot, name='chatbot'),
    # Auth
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    # Password reset
    path('password-reset/', auth_views.PasswordResetView.as_view(template_name='core/password_reset.html'), name='password_reset'),
    path('password-reset/done/', auth_views.PasswordResetDoneView.as_view(template_name='core/password_reset_done.html'), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(template_name='core/password_reset_confirm.html'), name='password_reset_confirm'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(template_name='core/password_reset_complete.html'), name='password_reset_complete'),
    # Admin dashboard
    path('dashboard/', views.admin_dashboard, name='admin_dashboard'),
] 