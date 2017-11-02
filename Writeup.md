# **Use Deep Learning to Clone Driving Behavior** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/final-architecture.png "Final Architecture"
[image2]: ./examples/center-lane-driving.jpg "Center Lane Driving"
[image3]: ./examples/recovery-part-1.jpg "Recovery Image"
[image4]: ./examples/recovery-part-2.jpg "Recovery Image"
[image5]: ./examples/recovery-part-3.jpg "Recovery Image"
[image6]: ./examples/flipped-1.jpg "Normal Image"
[image7]: ./examples/flipped-2.jpg "Flipped Image"

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

### Required Files

#### Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* Writeup.md summarizing the results

### Quality of Code

#### Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing the following command:
```sh
python drive.py model.h5
```

#### Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The code uses a generator (model.py line 123) for memory-efficiency. Instead of storing the preprocessed data in memory all at once, using a generator we can pull pieces of the data and process them on the fly only when we need them, which is much more memory-efficient.

### Model Architecture and Training Strategy

#### An appropriate model architecture has been employed

The final model architecture (model.py lines 65-82) consists of a convolution neural network based on the architecture described in this [NVIDIA paper](https://arxiv.org/pdf/1604.07316.pdf).

The convolutional layers are designed to perform feature extraction. NVIDIA used strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel, and a non-strided convolution with a 3×3 kernel size in the final two convolutional layers.

The data is normalized in the model using a Keras lambda layer (model.py line 70). Due to irrelevant image content, the image data is cropped from the top by 70 pixels and from the bottom by 25 pixels (model.py line71). The model includes RELU layers to introduce nonlinearity (model.py lines 72-76).

The model summary including output shapes and number of parameters per layer is as follows:

| Layer           | Output Shape   | Param # |
|:----------------|:--------------:|--------:|
| Input           | 160x320x3      | 0       |
| Lambda          | 160x320x3      | 0       |
| Cropping2D      | 65x320x3       | 0       |
| Convolution2D   | 31x158x24      | 1824    |
| Convolution2D   | 14x77x36       | 21636   |
| Convolution2D   | 5x37x48        | 43248   |
| Convolution2D   | 3x35x64        | 27712   |
| Convolution2D   | 1x33x64        | 36928   |
| Flatten         | 2112           | 0       |
| Dense           | 100            | 211300  |
| Dense           | 50             | 5050    |
| Dense           | 10             | 510     |
| Dense           | 1              | 11      |

|:--------------------------|--------:|
| **Total params:**         | 348,219 |
| **Trainable params:**     | 348,219 |
| **Non-trainable params:** | 0       |

#### Attempts to reduce overfitting in the model

I decided not to modify the model by applying regularization techniques like Dropout or Max pooling. Instead, I decided to keep the number of training epochs low: only three epochs. In addition to that, the model was trained and validated on different data sets to ensure that the model was not overfitting ([model.py lines 108-116](model.py#L108-L116)).

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually ([model.py line 122](model.py#L122)).

#### Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and doing the same in the opposite direction.

### Architecture and Training Documentation

#### Solution Design Approach

The overall strategy for deriving a model architecture was to start with a well-known architecture found in scientifical papers. My first step was to use a convolution neural network model similar to the [NVIDIA architecture](https://arxiv.org/pdf/1604.07316.pdf). This architecture has already been used for End-to-End Deep Learning in Self-Driving Cars.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that the model had quiet balanced and low mean squared errors on the training set and on the validation set. This implied that the model was pretty well fitted.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I changed the color space of each image as follows:

```
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### Final Model Architecture



Here is a visualization of the final architecture:

![Final Architecture][image1]

#### 2. Attempts to reduce overfitting in the model

I decided not to modify the model by applying regularization techniques like Dropout or Max pooling.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center Lane Driving][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to navigate back onto the road. These images show what a recovery looks like starting from ... :

![Recovery Image][image3]
![Recovery Image][image4]
![Recovery Image][image5]

To augment the data sat, I also flipped images and angles thinking that this would double the dataset. For example, here is an image that has then been flipped:

![Normal Image][image6]
![Flipped Image][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3. I used an adam optimizer so that manually training the learning rate wasn't necessary.
