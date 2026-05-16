"""
URL configuration for face_attendance project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.views.generic import RedirectView

urlpatterns = [
    path('django-admin/', admin.site.urls),  # Django built-in admin (đổi để tránh xung đột)

    path('', include('attendance.urls')),

    path('attendance/api/', include('attendance.urls_api')),

    # Redirect /sw.js to the actual static file so browser stops getting 404
    path('sw.js', RedirectView.as_view(
        url='/static/attendance/js/sw.js', permanent=False
    )),
]

urlpatterns += staticfiles_urlpatterns()
