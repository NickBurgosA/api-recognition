import cv2
import numpy as np
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from .model import load_model

def reconocer_imagen(src_image):
    image_size = 28
    batch_size = 256
    epochs = 20
    save_file = 'model.h5'
    input_shape = (image_size, image_size, 1)
    num_classes = 4
    class_names = ['basketball', 'airplane', 'banana', 'apple']

    model = load_model(input_shape, num_classes)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.load_weights(os.path.join(BASE_DIR,'api/',save_file))

    image = cv2.imread(src_image)
    image = cv2.resize(image, (image_size, image_size))
    image = cv2.Canny(image, 100, 200)
    image = image / 255
    image = np.array(image).reshape(1, image_size, image_size, 1).astype('float32')

    idx = model.predict_classes(image)[0]
    resultado = class_names[idx] if class_names[idx] is not None else 'No se encontró'

    return resultado
