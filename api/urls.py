from django.conf.urls import include, url
from rest_framework import routers
from .views import ImageViewSet

router = routers.SimpleRouter()
router.register(r'images', ImageViewSet)

urlpatterns = [
    url(r'^api/', include(router.urls, namespace='api')),
]