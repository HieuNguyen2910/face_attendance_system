# from django.urls import path
# from . import views

# urlpatterns = [
#     path('recognize/', views.api_recognize, name='api_recognize'),
#     path('register/', views.api_register, name='api_register'),
#     path('checkin/', views.api_checkin, name='api_checkin'),
#     path('checkout/', views.api_checkout, name='api_checkout'),  # <-- thêm check-out
#     path('checkin_status/', views.api_checkin_status, name='api_checkin_status'),
#     path('users/', views.api_list_users, name='api_list_users'),
#     path('delete_user/', views.api_delete_user, name='api_delete_user'),
#     path('history/', views.api_history, name='api_history'),
#     path('update_user/', views.api_update_user, name='api_update_user'),
#     path('replace_face/', views.api_replace_face, name='api_replace_face'),
#     path('register_employee/', views.api_register, name='api_register_employee'),

# ]

from django.urls import path
from . import views

urlpatterns = [
    path('recognize/', views.api_recognize, name='api_recognize'),
    path('register/', views.api_register, name='api_register'),
    path('checkin/', views.api_checkin, name='api_checkin'),
    path('checkout/', views.api_checkout, name='api_checkout'),
    path('checkin_status/', views.api_checkin_status, name='api_checkin_status'),
    path('users/', views.api_list_users, name='api_list_users'),
    path('update_user/', views.api_update_user, name='api_update_user'),
    path('delete_user/', views.api_delete_user, name='api_delete_user'),
    path('replace_face/', views.api_replace_face, name='api_replace_face'),
    path('register_employee/', views.api_register_employee, name='api_register_employee'),
    path('check_history/', views.api_history, name='api_history'),
    path('check_history/day/', views.api_history_by_day, name='api_history_by_day'),
    path('check_history/user/<str:user_id>/', views.api_history_by_id, name='api_history_by_id'),
    path('check_user/<str:user_id>/', views.api_check_user, name='api_check_user'),
]
