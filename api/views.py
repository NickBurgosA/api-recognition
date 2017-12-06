from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.response import Response
from .models import *
from .serializers import ImageSerializer
from .recognition import reconocer_imagen
from django.conf import settings
import os
# Create your views here.


class ImageViewSet(viewsets.ModelViewSet):
    queryset = ImageModel.objects.all();
    serializer_class = ImageSerializer

    def create(self, request, *args, **kwargs):
        image_serializer = ImageSerializer(data=request.data)
        image_serializer.is_valid()
        image_serializer.save()
        image = image_serializer.data.get('image')
        image = settings.MEDIA_ROOT+image[7:]
        respuesta = reconocer_imagen(image)

        return Response({"respuesta": respuesta}, status= status.HTTP_200_OK)