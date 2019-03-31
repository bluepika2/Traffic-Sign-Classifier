# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./output_images/y_label.png "y_label"
[image2]: ./output_images/grayscale.png "Grayscaling"
[image3]: ./NewTrafficSign/trafficsign4.png "Traffic Sign 4"
[image4]: ./NewTrafficSign/trafficsign1.png "Traffic Sign 1"
[image5]: ./NewTrafficSign/trafficsign2.png "Traffic Sign 2"
[image6]: ./NewTrafficSign/trafficsign5.png "Traffic Sign 5"
[image7]: ./NewTrafficSign/trafficsign3.png "Traffic Sign 3"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is len(X_train) --> 34799
* The size of the validation set is len(X_valid) --> 4410
* The size of test set is len(X_test) --> 12630
* The shape of a traffic sign image is X_train[0].shape --> (32, 32, 3)
* The number of unique classes/labels in the data set is len(pd.Series(y_train).unique().tolist()) --> 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing label frequency of y_train data.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it will be enough to classify traffic sign based on reference paper, and it will also help to decrease required memory and computational power.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because the image data should be normalized so that the data has mean zero and equal variance.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 3x3	    | 1x1 stride, VALID padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| flatten				| outputs 400									|
| Fully Connected		| outputs 120  									|
| RELU					|												|
| Dropout				|keep_prob = 0.5								|
| Fully Connected		| outputs 84  									|
| RELU					|												|
| Dropout				|keep_prob = 0.5								|
| Fully Connected		| outputs 43  									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an 100 for batch size, which can improve accruacy rather than 128 value.
For epochs, I choose 10 because it looks enough after I tried several options for epoch based on my model. If epoch value is too large, model accuracy will decrease, which cause overfitting.
I selected 0.0009 for learning rate. When I changed this value from 0.001 to 0.0009, I could see some improvement for my model. As far as I understood, decrease of learning rate should be able to improve accuracy, but it can also bring about more computational effort.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1
* validation set accuracy of 0.941 
* test set accuracy of 0.911

If a well known architecture was chosen:
* What architecture was chosen?
I basically utilized LeNet architecture which I learned from lessons.
* Why did you believe it would be relevant to the traffic sign application?
Based on the lessons, LeNet architecture really works well on image classification because it includes CNNs with fully connected networks.
CNN can detect special feature on images as it proceeds to next layer. However, this LeNet architecutre was not enough to meet 0.93 validation accuracy.
Right after I checked this accuracy, I came up with several methods including change of learning rate, epoch, batch size, and also LeNet architecture.
Decreaing learning rate will give better accuracy definitely, I should not reduce this value too much to consider computational time in real world.
A little decrease of batch size gives me better accuracy as well because model will update more frequently.
I also added Dropout to existing LeNet architecture to prevent overfitting.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep right      		| Keep right   									| 
| Right-of-way 			| Right-of-way 									|
| Speed limit (30km/h)	| Speed limit (30km/h)							|
| Road work				| Road work					 					|
| Priority road			| Priority road									|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 15th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a keep right sign (probability of 1.0), and the image does contain a keep right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Keep right   									| 
| 0.00     				| Turn left 									|
| 0.00					| Go straight or right							|
| 0.00	      			| Dangerous curve to the right	 				|
| 0.00				    | General caution      							|


For the second image, the model is relatively sure that this is a right-of-way sign (probability of 0.99), and the image does contain a right-of-way sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99        			| Right-of-way at the next intersection   		| 
| 0.00     				| Beware of ice/snow 							|
| 0.00					| Children crossing								|
| 0.00	      			| Slippery road	 								|
| 0.00				    | Double curve      							|

For the third image, the model is relatively sure that this is a Speed limit (30km/h) sign (probability of 0.88), and the image does contain a Speed limit (30km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.88        			| Speed limit (30km/h)   						| 
| 0.11     				| Speed limit (50km/h) 							|
| 0.00					| Speed limit (80km/h)							|
| 0.00	      			| Speed limit (100km/h)	 						|
| 0.00				    | Speed limit (20km/h)     						|

For the fourth image, the model is relatively sure that this is a Road work sign (probability of 0.47), and the image does contain a Road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.47        			| Road work (30km/h)   							| 
| 0.22     				| Bumpy road 									|
| 0.12					| Wild animals crossing							|
| 0.01	      			| Double curve	 								|
| 0.00				    | Keep left     								|

For the fifth image, the model is relatively sure that this is a Priority roadsign (probability of 0.47), and the image does contain a Priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        						| 
|:---------------------:|:-------------------------------------------------:| 
| 1.00        			| Priority road   									| 
| 0.00     				| Roundabout mandatory 								|
| 0.00					| End of no passing by vehicles over 3.5 metric tons|
| 0.00	      			| Speed limit (100km/h)	 							|
| 0.00				    | Keep right     									|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


