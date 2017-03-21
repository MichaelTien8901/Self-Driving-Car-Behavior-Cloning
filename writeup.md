**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Data Preprocessing ##
* Data Augmentation

   To deal with left turn bias from the training data, flipping the image to opposite side with negative measurement 
   is a effective method.
   
   ```python
  import numpy as np
  image_flipped = np.fliplr(image)
  measurement_flipped = -measurement
   ```
* Use Multiple Camera Data
   
   ```python
   correction = 0.2 # this is a parameter to tune
   steering_left = steering_center + correction
   steering_right = steering_center - correction
   ```
   
* Image Cropping

   This is built in the CNN using keras
   ```python
   model.add(Cropping2D(cropping=((60, 25), (0,0))))
   ```
* Value Normalization

  This is built in the CNN using keras
  ```python
  model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
  ```

## Convolution Neural Network ###

* NVIDIA model 

I borrowed the CNN from NVIDIA paper
[The deep learning self-driving car paper from NVIDIA](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)
to implment the behavior cloning project.  

```python)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
### cropping
model.add(Cropping2D(cropping=((60, 25), (0,0))))

model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
```

## Training and Validation ###
* Epoch number and Overfitting

   From the training and validation errors, I found out more epoch might cause higher validation errors, even the training error 
   is getting smaller.  It might be the sign of overfitting because the validation error isn't going down.  So epoch number = 3 
   is chosen for most of the test.
   
   ```python
   model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
   ```

## Test Result on Track 1 ##
Auto driving in track 1 is pretty stable.  There is not obvious "drunk driving" effect for this model. 
Set speed 9, 15, and 25 are almost the same result.  

## Various Test
* Different Color Space

I've tried to convert RGB color space to HSV hoping to deal with the brightness difference problem.  For the current
track and data collected, no obvious advantage found.

* Greyscale with Histogram Equalization

I tried to enhance the image using the histogram equalization.  The cv2 function only worked for greyscale image.  

```python
def preprocessing_image(image):
    out_image = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return out_image
```

The result model can't find color different between road and grass and can't finish the running track.

* Unstable Result for Same Train Set

The same training data set might generate different result for different run of the model.y.  This might be result of too little 
training data.  The split of validation and training data might cause some features not trained well.  

* Adaptive Correction Angle for Left and Right Image

   In stead of using fix correction angle for left and right images, an adaptive angle formula is used like
   ```python
   CORRECTION_ANGLE = FIX_ANGLE * (1 + center_steering / MAX_STEERING)
   ```
   I didn't find so much difference in terms of stability from the fix correction angle approach.  

* Driving Speed

The intuition is to slow down when turning for human driver.  The response time for steering
is so fast for this simulator that it seems no difference for speed 9, 15, and 25. This might be difference 
for real car driving on road.



