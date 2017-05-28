# **Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/train_dist.png "Training Data Distribution"
[image11]: ./examples/test_dist.png "Testing Data Distribution"
[image12]: ./examples/valid_dist.png "Validation Data Distribution"
[image13]: ./examples/img_ex.png "Examples of Images in dataset"
[image2]: ./examples/exp.png "Explaination of Wrong result"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image14]: ./examples/w1.png "W1"
[image15]: ./examples/w2.png "W2"

[image4]: ./examples/1.jpg "Traffic Sign 1"
[image5]: ./examples/2.jpg "Traffic Sign 2"
[image6]: ./examples/3.jpg "Traffic Sign 3"
[image7]: ./examples/4.jpg "Traffic Sign 4"
[image8]: ./examples/5.jpg "Traffic Sign 5"
[image9]: ./examples/6.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/himanshubabal/Traffic-Signs/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the 6th code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the 7th and 9-11th code cells of the IPython notebook.  

Here is an exploratory visualization of the data set.

It shows images from datasets to give us a visual how the data looks

![alt text][image13]


Below are bar charts showing the distribution of labels in the dataset.

![alt text][image1]


![alt text][image11]


![alt text][image12]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

In this project, I have not done any preprocessing because I was on a CPU machine and training a model takes up a lot of time.

Although I have some ideas that I would like to try on images
* Normalizing
* Mean - centering
* Color conversion, etc


#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I used datasets provided by Udacity.
It had training, testing and validation datasets.

* The size of training set is 34799
* The size of test set is 12630


My final training set had 37499 number of images.

In this project, I have not done any augmentation to data because I was on a CPU machine and training a model takes up a lot of time.

Although I have some ideas that I would like to try on images
* Tilting image 15 degrees on either side to introduce invariance to tilt
* Flip image horizontally
* Change in brightness of image
* Adding noise in the image
* Removing some portion of image at random to make it better at recognizing occluded images

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located `LeNet()` method of the ipython notebook.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, 10x10x16      
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
|Flatten|output - 400|
| Fully connected		|        Input - 400, Output 1024 			
|ReLU||
|Dropout| pkeep - 50%|
| Fully connected		|        Input - 1024, Output 512
|ReLU||
|Dropout| pkeep - 50%|
| Softmax				| 43 outputs       									|




#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Code - `In[38] - In[42]` of Ipython Notebook.

To train the model, I used tensorflow to train my model.

* Optimizer - Adam Optimizer
* Learning Rate - 0.001
* Epochs - 50
* Batch Size - 128

Initially, I trained my model for 5 epochs as I was on CPU machine and it was taking a long time to train.
But then accuracy was not good, so I decided to train it for 50 epochs in order to get better results.



#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of - 98 %
* validation set accuracy of - 95.2 %
* test set accuracy of - 93.4 %

If an iterative approach was chosen:
* Initial Model - Simple CNN with 1 convolutional and 1 fully connected layer. Performed not-so-well, so decided to increase number of layers.
* Increased layers to 2 Conv and 2 FC layers. Still results were not so good.
* Added 2 Max Pool layer. Achieved decent accuracy, yet wanted to do bit batter.
* Added Dropout layers after FC layers with 50% dropout. Achieved high accuracy on the data. :)


* What were some problems with the initial architecture?
    * Initial Architecture was not deep enough, and thus was not presenting with expected accuracy.


* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    * As described above, I added More Conv, FC, Max Pool and Dropout layers.


* Which parameters were tuned? How were they adjusted and why?
    * Dropout pkeep - decreased from 0.70 to 0.50
    * Increased number of neurons from 500 to 1024.
    * decreased Learning rate from 0.01 to 0.001


* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    * Convolutional Layers in some sense preserve information in image which otherwise is destroyed by Fully connected layers.
    * Max Pool layers also help in preserving this info while decreasing size.
    * My model was quite similar to `LeNet`, but with some tweaks, which I think made it work better

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
    * training set accuracy of - 98 %
    * validation set accuracy of - 95.2 %
    * test set accuracy of - 93.4 %

    * As I have received very high accuracy on all of the datasets, it suggests that my model should be working very well and shows no sign of overfitting.

    * Also, I have tested it on other images downloaded from Internet and it predicted quite well, thus I am pretty sure model is working as intended.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4]

![alt text][image5]

![alt text][image6]

![alt text][image7]

![alt text][image8]

![alt text][image9]

* Images 1, 2, 5 and 6 should be very easy to classify as these are picked up directly from wikipedia.

* Image 3 should also be kinda easy to classify as matches closely to the template.

* Image 4 should be probably very tough to classify as it contains image in a rather smaller portion of image and it also contains an extra rectangular sign on it.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Slippery Road | Slippery Road	|
| 	Traffic Signals |	Traffic Signals|
| Speed Limit - 60 kmph|Speed Limit - 60 kmph|
| 	Speed Limit - 30 kmph| Priority Road|
| General caution	| General caution|
|Dangerous curve to the left|Dangerous curve to the left|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.33%. This compares favorably to the accuracy on the test set of 93.4%

##### Explaining wrong result

I believe that image was recognized wrong because of the compression of image down to 32x32.
After compressing the image to that resolution, even humans can not recognise it.

I believe if image was taken from a close up, it could have been easily recognised.

It also somewhat matches the image the model predicted as shown in image below.

![alt text][image2]

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the `get_top5()` method Ipython notebook.

For the first image, the model is 100% sure that sign is slippery road . The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| Slippery Road |
| 0.0     				| Speed limit (20km/h)	|
| 0.0					| Speed limit (30km/h)	|
| 0.0      			    | Speed limit (40km/h)	|
| 0.0				    | Speed limit (50km/h)	|



For the second image

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| Traffic Signals |
| 0.0     				| Speed limit (80km/h)	|
| 0.0					| Speed limit (50km/h)	|
| 0.0      			    | Speed limit (20km/h)	|
| 0.0				    | Speed limit (30km/h)	|

For the third image


| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| Speed limit (60km/h) |
| 1.5e-32     				| Speed limit (80km/h)	|
| 2.8e-37				| Speed limit (50km/h)	|
| 0.0      			    | Speed limit (20km/h)	|
| 0.0				    | Speed limit (30km/h)	|

For the fourth image

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.999        			| Priority Road |
| 3.5e-7    				| Right-of-way at the next intersection	|
| 9.0e-8				| End of speed limit (80km/h)	|
| 5.0e-8      			    | Double curve	|
| 3.0e-8				    | No passing for vehicles over 3.5 metric tons	|

For the fifth image

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| General caution |
| 0.0     				| Speed limit (20km/h)	|
| 0.0					| Speed limit (30km/h)	|
| 0.0      			    | Speed limit (40km/h)	|
| 0.0				    | Speed limit (50km/h)	|


#### Visualizing the weights

I also visualized weights of learned model

Weights of 1st Convolutional layer (6 outputs)

![alt text][image14]

Weights of 2nd Convolutional layer (16 outputs)

![alt text][image15]
