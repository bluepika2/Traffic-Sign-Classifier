## Project: Build a Traffic Sign Classifier with Deep Neural Networks

Overview
---
In this project, I will use what I've learned about deep neural networks and convolutional neural networks to classify traffic signs. I will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, I will then try out my model on images of German traffic signs that I find on the web, which will be new test set.


The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results

[//]: # (Image References)

[image1]: ./output_images/dataset-distributions_project3.png "distributions"
[image2]: ./output_images/grayscale.png "Grayscaling"
[image3]: ./NewTrafficSign/trafficsign4.png "Traffic Sign 4"
[image4]: ./NewTrafficSign/trafficsign1.png "Traffic Sign 1"
[image5]: ./NewTrafficSign/trafficsign2.png "Traffic Sign 2"
[image6]: ./NewTrafficSign/trafficsign5.png "Traffic Sign 5"
[image7]: ./NewTrafficSign/trafficsign3.png "Traffic Sign 3"
[image8]: ./output_images/original_image.png "original"
[image9]: ./output_images/conv1_image.png "conv1"
[image10]: ./output_images/conv2_image.png "conv2"

---

### Data Set Summary & Exploration

#### 1. Basic identification of data set

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is len(X_train) --> 34799
* The size of the validation set is len(X_valid) --> 4410
* The size of test set is len(X_test) --> 12630
* The shape of a traffic sign image is X_train[0].shape --> (32, 32, 3)
* The number of unique classes/labels in the data set is len(pd.Series(y_train).unique().tolist()) --> 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram chart showing distribution of training, testing and validation data set.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Description of image data processing techniques 

As a first step, I decided to convert the images to grayscale because it will be enough to classify traffic sign based on its edges & shapes, and it will also help to decrease required memory and computational power.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because the image data should be normalized so that the data has mean zero and equal variance.



#### 2. Descriptions of my final model architecture for deep neural networks.

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
 


#### 3. Hyperpapameters decision and assumption for model training

To train the model, I used an 100 for batch size, which can improve accruacy rather than 128 value, and also can save some memory.
For epochs, I choose 10 because it looks enough after I tried several options for epoch based on my model. If epoch value is too large, model accuracy will decrease, which cause overfitting.
I selected 0.0009 for learning rate. When I changed this value from 0.001 to 0.0009, I could see some improvement for my model. As far as I understood, decrease of learning rate should be able to improve accuracy, but it can also bring about more computational effort.

#### 4. Discussion on approach for finding a solution and getting better validation set accuracy

My final model results were:
* training set accuracy of 1
* validation set accuracy of 0.941 
* test set accuracy of 0.911

* What architecture was chosen?
I basically utilized LeNet architecture.
* Why did you believe it would be relevant to the traffic sign application?
LeNet architecture really works well on image classification because it includes CNNs with fully connected networks.
CNN can detect special feature on images as it proceeds to next layer. However, this LeNet architecutre was not enough to meet 0.93 validation accuracy.
Right after I checked this accuracy, I came up with several methods including change of learning rate, epoch, batch size, and also LeNet architecture.
Decreaing learning rate will give better accuracy definitely, I should not reduce this value too much to consider computational time in real world.
A little decrease of batch size gives me better accuracy as well because model will update more frequently.
I also added Dropout to existing LeNet architecture to prevent overfitting.

### Visualizing the Neural Network
While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.

For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. 
I tried experimenting with a similar test to show that my trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.
Let's see original image first:

![alt text][image8]

Now we can look at how that image activates the neurons of the first convolutional layer. Notice how each filter has learned to activate optimally for different features of the image.

![alt text][image9]

Also, we can check the second convolutional layer too.

![alt text][image10]


### Test a Model on New Images

#### 1. New five German traffic signs found on the web 

Here are five German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7]


#### 2. The model's predictions on these new traffic signs and compare the results to predicting on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep right      		| Keep right   									| 
| Right-of-way 			| Right-of-way 									|
| Speed limit (30km/h)	| Speed limit (30km/h)							|
| Road work				| Road work					 					|
| Priority road			| Priority road									|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

The code for making predictions on my final model is located in the 15th cell of the `Traffic_Sign_Classifier.ipynb`.

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
