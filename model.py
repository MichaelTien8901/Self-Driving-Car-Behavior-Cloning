# Load data
import cv2
import csv
import numpy as np
# from keras.utils.visualize_util import plot
import matplotlib.pyplot as plt

def processing(x):
    return x

## for unix, path splitter is /
## for windows, path splitter is \
PATH_SEPARATOR = '\\'
# PATH_SEPARATOR = '/'
STEERING_CORRECTION = 0.3
EPOCH_NO = 3
SAVE_MODEL_NAME = 'model.h5'

data_dirs = ['data1', 'data2', 'data3', 'data4']
# data_dirs = ['data']
#
# collect data from data directories
images = []
measurements = []
for i in range(len(data_dirs)):
    lines = []
    with open(data_dirs[i] + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    for line in lines:
        steering_center = float(line[3])
        steering_left = steering_center + STEERING_CORRECTION
        steering_right = steering_center - STEERING_CORRECTION

        center_filename = line[0].split(PATH_SEPARATOR)[-1]
        left_filename = line[1].split(PATH_SEPARATOR)[-1]
        right_filename = line[2].split(PATH_SEPARATOR)[-1]
        current_path = data_dirs[i] + PATH_SEPARATOR + 'IMG' + PATH_SEPARATOR
        center_image = processing(cv2.imread(current_path + center_filename))
        left_image = processing(cv2.imread(current_path + left_filename))
        right_image = processing(cv2.imread(current_path + right_filename))
        images.extend([center_image, left_image, right_image])
        measurements.extend([steering_center, steering_left, steering_right])

#
# Data Augmentation with flipping side of images
#

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    if measurement == 0:
        continue
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1.0)
#   ---------------------------------------------------
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
#
# use keras to build NVIDIA Model
#
DROPOUT_RATE = 0.5
#
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
##from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
### cropping
model.add(Cropping2D(cropping=((70, 25), (24,24))))

model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Dropout(DROPOUT_RATE))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Dropout(DROPOUT_RATE))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Dropout(DROPOUT_RATE))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Dropout(DROPOUT_RATE))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Dropout(DROPOUT_RATE))

model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(DROPOUT_RATE))
model.add(Dense(50))
model.add(Dropout(DROPOUT_RATE))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
# plot(model, to_file='model.png')

history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=EPOCH_NO)
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=EPOCH_NO)

model.save(SAVE_MODEL_NAME)
print("model saved as %s" % SAVE_MODEL_NAME)

# print(history_object.history.keys())

#
# plot the training and validation loss for each epoch
#
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
