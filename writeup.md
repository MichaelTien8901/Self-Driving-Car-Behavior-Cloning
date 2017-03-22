**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report
## Problems During Training ## 
* Left Turn Bias

  At the begining, the left turn bias is very obviously troublesome.  Use data augmentation with flipping side of images 
  solve this problem immdiately.
  
* Very Large Training and Validation Loss

  After analyzing images, I cropping out the left and right side by 24 pixels, top by 60 pixels, and bottom by 24 pixels.  
  The loss shrink a lot and the auto drive in simulator seems to work.
  
* Can't Keep in the Middle of Lane
  
  Use multiple camera images with 'correction angle', the car seems to know how to keep in the middle of lane.
  
* Sharp Turn Failure

  The car failed in some sharp turns.  After add more training data of these turns, the car still can't stably keep in the middle 
  of the turns. The other way is tuning the 'correction angle' used for multiple camera images.  After increase the correction angle
  a bit, the car finally turn successfully.

* Can't Differentiate Lane and Grass

  At some turns where no lane mark at one side, the car can't keep inside the lane.  Don't know it is because the sharp turn or 
  the lane recognition. I add more training data in these places.  Combined with correction angle value, this problem seems disappear.

* Overfitting 
   
   From the training and validation loss chart, it seems overfitting when validation loss didn't decrease with training loss.  I 
   put the dropout layers into the model to prevent overfitting. 
   
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
   model.add(Cropping2D(cropping=((60, 25), (24,24))))
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

```python
DROPOUT_RATE = 0.5
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

```

|Layers       |  Dimension |
|-------------|------------|
|Input Layer  | 3@160x320  |
|Normalization| 3@160x320  |
|Cropping     |  3@75x272  |
|5x5 Convolutional  | 24@38x136 |
|0.5 Dropout        |           |
|5x5 Convolutional  | 36@19x68  |
|0.5 Dropout        |          |
|5x5 Convolutional  | 48@10x34  |
|0.5 Dropout        |          |
|3x3 Convolutional  | 64@8x32   |
|0.5 Dropout        |          |
|3x3 Convolutional  | 64@6x30   |
|0.5 Dropout        |          |
|Flatten            | 11520     |
|Full Connected     | 100       |
|0.5 Dropout        |          |
|Full Connected     | 50        |
|0.5 Dropout        |          |
|Full Connected     | 10   |
|Full Connected     | 1   |

## Training and Validation ###
* Epoch number and Overfitting

   From the training and validation errors, I found out more epoch might cause higher validation errors, even the training error 
   is getting smaller.  It might be the sign of overfitting because the validation error isn't going down.  
   
 
![Overfitting](https://github.com/MichaelTien8901/Self-Driving-Car-Behavior-Cloning/blob/master/overfitting.png "Overfitting if Epoch > 2")
   
   So epoch number = 3 is chosen for most of the test.
      
   ```python
   model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
   ```
| Training Loss | Validation Loss  |
| -------------:| ----------------:|
| 0.0306        |  0.0405          |

![Training Result](https://github.com/MichaelTien8901/Self-Driving-Car-Behavior-Cloning/blob/master/training_loss.png "Training Loss")

## Test Result on Track 1 ##

Auto driving in track 1 is pretty stable.  There is not obvious "drunk driving" effect for this model. 
Set speed 9, 15, and 25 are almost the same result.  

## Various Test

* Different Color Space

I've tried to convert RGB color space to HSV hoping to deal with the brightness difference problem.  

```python
def preprocessing_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
```

For the current track and data collected, no obvious advantage found.

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



