
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='index'),
    path('manage/', views.manage, name='manage'),
    path('history/', views.history, name='history'),
    path('history/id/<str:user_id>/', views.history_by_id, name='history_by_id'),
    
]
