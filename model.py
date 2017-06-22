import csv

import cv2
import numpy as np


left_imgs = []
center_imgs = []
right_imgs = []
steerings = []
brakes = []

with open('driving_log.csv') as f:
    reader = csv.reader(f)
    for line in reader:
        imgs = list(map(cv2.imread, line[:3]))
        steering = float(line[3])
        throttle = float(line[4])
        brake = float(line[5])
        speed = float(line[6])

        center_imgs.append(imgs[0])
        left_imgs.append(imgs[1])
        right_imgs.append(imgs[2])
        steerings.append(steering)
        brakes.append(brake)


def invert_data(X, y):
    inverted_X = X[:, :, ::-1, :]
    inverted_y = -y
    return inverted_X, inverted_y


class TrainingBatch(object):
    """docstring for TrainingBatch"""

    def __init__(self, X_train, y_train, validation_split, batch_size):
        split_num = int((1 - validation_split) * len(X_train))
        self.X_train = X_train[:split_num]
        self.y_train = y_train[:split_num]
        self.X_val = X_train[split_num:]
        self.y_val = y_train[split_num:]
        self.batch_size = batch_size
        self.steps_per_epoch = split_num / batch_size

    def get_validation_data(self):
        return self.X_val, self.y_val

    def get_steps_per_epoch(self):
        return self.steps_per_epoch

    def generator(self):
        while True:
            indices = np.random.choice(len(self.X_train),
                                self.batch_size, replace=False)
            yield (self.X_train[indices], self.y_train[indices])


X_train = np.array(center_imgs)
y_train = np.array(steerings)

correction_factor = 0.15

X_left_train = np.array(left_imgs)
y_left_train = y_train + correction_factor

X_right_train = np.array(right_imgs)
y_right_train = y_train - correction_factor


inverted_X_train, inverted_y_train = invert_data(X_train, y_train)

X_train = np.concatenate([X_train, inverted_X_train], axis=0)
y_train = np.concatenate([y_train, inverted_y_train], axis=0)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

batch_size = 128
training_batch = TrainingBatch(X_train, y_train, 0.2, batch_size)

model = Sequential()
model.add(Lambda(lambda x: x / 255. - .5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(1, 1), activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.fit_generator(training_batch.generator(),
                    training_batch.get_steps_per_epoch(),
                    nb_epoch=20,
                    validation_data=training_batch.get_validation_data())

model.save('model.h5')
