import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss

from keras import backend as K
from keras.utils import to_categorical

from model import load_model

image_size = 28
batch_size = 256
epochs = 20
save_file = 'model.h5'
input_shape = (image_size, image_size, 1)
num_classes = 4
class_names = ['basketball', 'airplane', 'banana', 'apple']


def load_data(root, vfold_ratio=0.2, max_items_per_class=10000):
    all_files = glob.glob(os.path.join(root, '*.npy'))

    x = np.empty([0, 784])
    y = np.empty([0])
    class_names = []

    for idx, file in enumerate(all_files):
        data = np.load(file)
        data = data[0:max_items_per_class, :]
        labels = np.full(data.shape[0], idx)

        x = np.concatenate((x, data), axis=0)
        y = np.append(y, labels)

        class_name, ext = os.path.splitext(os.path.basename(file))
        class_names.append(class_name)

    data = None
    labels = None
    permutation = np.random.permutation(y.shape[0])

    x = x[permutation, :]
    y = y[permutation]

    vfold_size = int(x.shape[0] / 100 * (vfold_ratio * 100))

    x_test = x[0:vfold_size, :]
    y_test = y[0:vfold_size]

    x_train = x[vfold_size:x.shape[0], :]
    y_train = y[vfold_size:y.shape[0]]

    return x_train, y_train, x_test, y_test, class_names


def visualize(X, Y, classes, samples_per_class=10):
    nb_classes = len(classes)

    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(Y == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)

        for i, idx in enumerate(idxs):
            plt_idx = i * nb_classes + y + 1
            plt.subplot(samples_per_class, nb_classes, plt_idx)
            plt.imshow(X[idx], cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    # plt.show()
    plt.savefig('preview/data.png')
    plt.clf()


def reconocer_imagen(src_image):
    model = load_model(input_shape, num_classes)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.load_weights(save_file)
    #    score, acc = model.evaluate(x_test, y_test, verbose=0)
    #   print('Test score:', score)
    #   print('Test accuracy:', acc)

    image = cv2.imread(src_image, 0)
    image = cv2.resize(image, (image_size, image_size))
    image = cv2.Canny(image, 100, 200)
    image = image / 255
    image = np.array(image).reshape(1, image_size, image_size, 1).astype('float32')

    idx = model.predict_classes(image)[0]
    resultado = class_names[idx] if class_names[idx] is not None else 'No se encontró'

    return resultado
