from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r"^ws/camera-display/$", consumers.CameraDisplayConsumer.as_asgi()),
]
