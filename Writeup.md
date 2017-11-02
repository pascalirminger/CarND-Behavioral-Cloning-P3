# **Use Deep Learning to Clone Driving Behavior** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/nvidia-cnn-architecture.png "NVIDIA architecture"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

### Required Files

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* Writeup.md summarizing the results

### Quality of Code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The code uses a generator (model.py line 123) for memory-efficiency. Instead of storing the preprocessed data in memory all at once, using a generator we can pull pieces of the data and process them on the fly only when we need them, which is much more memory-efficient.

### Model Architecture and Training Strategy

The final model architecture (model.py lines 65-82) consists of a convolution neural network based on the architecture described in this [NVIDIA paper](https://arxiv.org/pdf/1604.07316.pdf).

The convolutional layers are designed to perform feature extraction. NVIDIA used strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel, and a non-strided convolution with a 3×3 kernel size in the final two convolutional layers.

The data is normalized in the model using a Keras lambda layer (model.py line 70). Due to irrelevant image content, the image data is cropped from the top by 70 pixels and from the bottom by 25 pixels (model.py line71). The model includes RELU layers to introduce nonlinearity (model.py lines 72-76). 

####2. Attempts to reduce overfitting in the model

=== The model contains dropout layers in order to reduce overfitting (model.py lines 21).  ===

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 109). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 122).

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and doing the same in the opposite direction.

=== For details about how I created the training data, see the next section. ===

### Architecture and Training Documentation

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a well-known architecture found in scientifical papers.

My first step was to use a convolution neural network model similar to the [NVIDIA architecture](https://arxiv.org/pdf/1604.07316.pdf). This architecture has already been used for End-to-End Deep Learning in Self-Driving Cars.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that the model had quiet balanced and low mean squared errors on the training set and on the validation set. This implied that the model was pretty well fitted.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I changed the color space of each image as follows:

```
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 65-82) consisted of a convolution neural network based on the architecture described in this [NVIDIA paper](https://arxiv.org/pdf/1604.07316.pdf). It was built with the following layers and layer sizes ...

Here is a visualization of the NVIDIA architecture:

![NVIDIA architecture][image1]
Source: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data sat, I also flipped images and angles thinking that this would double the dataset. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3. I used an adam optimizer so that manually training the learning rate wasn't necessary.
