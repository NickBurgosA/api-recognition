from django.conf.urls import include, url
from rest_framework import routers
from .views import ImageViewSet
from . import views

router = routers.SimpleRouter()
router.register(r'images', ImageViewSet)

urlpatterns = [
    url(r'^api/', include(router.urls, namespace='api')),
    url(r'^$', views.home, name='home'),
]