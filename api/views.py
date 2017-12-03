from django.shortcuts import render
from rest_framework import viewsets
from .models import *
from .serializers import ImageSerializer

# Create your views here.

class ImageViewSet(viewsets.ModelViewSet):
    queryset = ImageModel.objects.all();
    serializer_class = ImageSerializer