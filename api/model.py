from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import MaxPooling2D, Conv2D

def load_model(input_shape, num_classes):
  model = Sequential()

  model.add(Conv2D(6, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding="same"))
  model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(64, kernel_size=(3, 3), border_mode='same', activation='relu'))
  model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(512, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation='softmax'))

  return model