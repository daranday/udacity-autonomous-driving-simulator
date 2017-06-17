#**Project: Behavioral Cloning for Self-Driving Car**

Overview
---
In this project we are using a convolutional neural network to drive a car autonomously in a simulated environment. We built a convolutional neural network in Keras that takes in an image and outputs a steering angle. We train the network by recording a human driver driving a vehicle on a track and we want the neural network to learn the human's steering response to the view of the road.  We split up the collected data into training and validation set, to avoid overfitting. Eventually we test the performance of the neural network by letting the neural network drive autonomously one lap around the simulated track without going into the edges or off the road.


[//]: # (Image References)

[image2]: ./writeup_figures/center.jpg "Center driving"
[image3]: ./writeup_figures/extreme.jpg "Steep turns"
[image4]: ./writeup_figures/recover1.jpg "Recover Right"
[image5]: ./writeup_figures/recover2.jpg "Recover Left"
[image6]: ./writeup_figures/difficult.jpg "Difficult"


---
###Files Submissions

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model is consisted of convolutional layers with filter sizes 5x5 and 3x3 with depths varying from 24 to 64, max-pooling layers, flattening layer and fully connected layers.

The convolutional layers extract the visual features from the images, the max-pooling layers reduces the dimensionality of the model and helps overfitting and the fully connected layer allows for linear regression on the steering angle from those features it detects.

The image data is normalized such that the r, g, b values rest in the range of [-0.5, 0.5] because large values of input saturates the network, having small values for input helps the backpropagation work efficiently. After that it is cropped to remove unuseful pixels in the top and bottom

####2. Attempts to reduce overfitting in the model

We split the data into training and validation set, and train and validate separately to make sure the validation error does not increase when training error keeps decreasing. I also used a maxpooling layer which reduces model complexity. I tried to use dropout layers but it seems to desensitive the model towards varying road conditions, so I ended up not using them.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

We chose the training dataset such that it teaches the car to stay in the center lane, and also to steer back towards the center when it veers off. We want to keep the latter portition large to avoid learning only small angle adjustments.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The strategy I used in developing the autonomous driving model is to improve architecture and training data incrementally.

My first step was to create a working neural network that could give steering signals. My first approach was to have one convolutional layer, a flattening layer and one fully connected layer. The training data I used was a video where the car stayed perfectly centered. I only used center frames. To test the model, I turned on the simulator in autonomous mode and saw how the car handles. The result was that the car kept swivelling and does not stay within the confines of the road and liked to get off the road on the left side.

Then I took the advice to normalize the data, mirror the input and output, remove the unused pixels and used the left and right frames with a correction factor to the steering angle. Now it follows the road a bit, but still too slow and not consistent.

I then used the more powerful Lenet-5 architecture, changing the top and bottom layers only to accomodate the input and output format. This time the car seems to follow the road for the first road conditions, but does not fare well with the more complex transitions or the bridge.

So I switched to the convolution architure used by the Nvidia autonomous car team which has five convolution with RELU activations and striding, replacing one convolution layer with maxpooling layer. I also changed my training data collection strategy. I started to intentionally veer off the center and record only when I try to recover to the center lane. This helped significantly to correct the car when it's running towards the border. I collect even more data for when the curb temporarily disappears, and that seemed to combat adverse road conditions more so than the architecture or data augmentation strategies.

Finally with the more comprehensive training images and complex architecture, the model is able to drive around the track one lap without running off the curb.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes ...

Input image size: 160x320x3
Layer 1: Custom transformation layer that normalizes the data into [-0.5, 0.5] for the rgb color space.
Layer 2: Cropping layer that removes top 70 and bottom 25 rows of pixels from the input image.
Layer 3: Convolution layer with 24 5x5 kernels and a stride of 2x2, with relu activation
Layer 4: Convolution layer with 36 5x5 kernels and a stride of 2x2, with relu activation
Layer 5: Convolution layer with 48 5x5 kernels and a stride of 2x2, with relu activation
Layer 6: MaxPooling layer with 2x2 stride.
Layer 7: Convolution layer with 64 3x3 kernels and a stride of 1, with relu activation
Layer 8: Convolution layer with 64 3x3 kernels and a stride of 1, with relu activation
Layer 9: Flattening layer
Layer 10: Fully connected layer with output size 100.
Layer 11: Fully connected layer with output size 50.
Layer 12: Fully connected layer with output size 10.
Layer 13: Fully connected layer with output size 1.

####3. Creation of the Training Set & Training Process

To capture the ideal driving behavior, I recorded the car driving in the center. When the turn is steep, and/or the curb disappears on one side, I stick closer to the other side to avoid the model from accidentally steering into the side.

Center driving:
![alt text][image2]

Steep turns:
![alt text][image3]

But small deviations in the road can cause the car to be close to the border, and to teach the car to recover, I also recorded driving back to the center from the sides:

![alt text][image4]
![alt text][image5]

I also took more data samples for short but difficult scenarios, e.g. curb disappears at a steep turn:

![alt text][image6]

For the training process, the data was shuffled to increase the randomness of the process to avoid the weights being skewed beyond repair unintentionally. The train/validation split was 20%. There are in total 66K images for training (including left, center and right). The data augmentation doubled that count, so our validation data is copious enough to be representative of the whole dataset.

One thing to note is that the absolute values of the training and validation errors don't matter. Since if you keep stay in center lane it's easy to train the model to follow that so the error is small, but if the car ended up somewhere else it's hard to correct, and the robustness of the model is actually less than if the training data is more diverse and result in larger overall errors.
