
from django.urls import path
from . import views, auth_views

urlpatterns = [
    # ===== AUTH =====
    path('login/', auth_views.login_view, name='login'),
    path('logout/', auth_views.logout_view, name='logout'),

    # ===== ADMIN =====
    path('admin/dashboard/', auth_views.admin_dashboard, name='admin_dashboard'),
    path('admin/manage/', auth_views.admin_manage_employees, name='admin_manage'),
    path('admin/create_admin/', auth_views.admin_create_admin, name='admin_create_admin'),
    path('admin/history/', auth_views.admin_view_history, name='admin_history'),
    path('admin/employee/<str:user_id>/history/', auth_views.admin_employee_history, name='admin_employee_history'),
    path('admin/checkin/', auth_views.admin_checkin_camera, name='admin_checkin'),
    path('admin/schedule/update/', auth_views.admin_update_schedule, name='admin_update_schedule'),

    # ===== EMPLOYEE =====
    path('employee/dashboard/', auth_views.employee_dashboard, name='employee_dashboard'),
    path('employee/checkin/', auth_views.employee_checkin, name='employee_checkin'),
    path('employee/history/', auth_views.employee_history, name='employee_history'),
    path('employee/change_password/', auth_views.employee_change_password, name='employee_change_password'),

    # ===== ROOT: redirect based on auth state =====
    path('', auth_views.root_redirect, name='index'),
    path('manage/', auth_views.admin_manage_employees, name='manage'),
    path('history/', auth_views.admin_view_history, name='history'),
    path('history/id/<str:user_id>/', views.history_by_id, name='history_by_id'),

]
