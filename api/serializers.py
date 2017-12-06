from rest_framework import serializers
from .models import ImageModel


class ImageSerializer(serializers.ModelSerializer):
    respuesta = serializers.CharField(read_only=True)

    class Meta:
        model = ImageModel
        fields = ('id', 'image', 'respuesta')
