# Deep-Learning-KNN-MLP-Fashion-MNIST
This repository contains the ipnyb for image classification on the Fashion-MNIST dataset based on KNN and MLP.

## Objective 1
### 1.0 Discussion on Parameter Settings for K-NN
Given the images in our test dataset, we want to find k images in the train dataset that are nearest to the test images, where the proximity is dependent on the choice of our distance metrics. For the purpose of this assignment, we will define our distance metrics as L1 / Manhattan Distance, L2 / Euclidean Distance and Cosine Similarity. We also need to train our K-NN models based on various k values to determine the k value that gives the best accuracy for each distance metrics. In this case, we will set our k to be a number in the range of 3 to 9.

### 1.1 Discussion on Results Obtained

| Distance Metric | Results Obtained for Each k Value |
| --- | --- |
| L1 / Manhattan Distance | ![A1-1](https://user-images.githubusercontent.com/50171205/59688456-51bd4000-9210-11e9-948a-c5db50ebbbc1.png) |
| L2 / Euclidean Distance | ![A1-2](https://user-images.githubusercontent.com/50171205/59688461-5255d680-9210-11e9-87cb-478c337cb9b0.png) |
| Cosine Similarity | ![A1-3](https://user-images.githubusercontent.com/50171205/59688464-53870380-9210-11e9-9a32-bddf7b90f92f.png) |

The top performing K-NN models for each distance metric are highlighted in red as above. To visualize the results obtained better, the accuracy rate and the time taken to train the model against the corresponding k value are shown in the plot below:

![A1-4](https://user-images.githubusercontent.com/50171205/59688465-541f9a00-9210-11e9-8b45-283daf5c47a0.png)

Even though all top performing models for each distance metric only have slight differences in performance, the best model is the L1 / Manhattan Distance trained model where k = 6, with an accuracy rate of 0.856.

## Objective 2a
### 2.0 Summary of Test Accuracy Results Obtained for Each MLP Model

| Hidden Units | 0 Hidden Layer | 1 Hidden Layer | 2 Hidden Layers | 5 Hidden Layers |
| --- | --- | --- | --- | --- |
| NIL | **0.854629** | - | - | - |
| 16 | - | 0.866743 | 0.856857 | 0.860286 |
| 64 | - | 0.8752 | 0.876857 | 0.879714 |
| 256 | - | 0.876286 | 0.888971 | **0.889829** |

### 2.1 Discussion on Results Obtained
#### 2.1.1 Influence of Number of Hidden Units on Test Accuracy 
Based on the results obtained, we can deduce that for the same number of hidden layers in a model, an increase in the number of hidden units leads to an increase in the model performance on test accuracy. In this case, the 5 hidden layers with 256 hidden units MLP model obtained the best accuracy (in green), while the 0 hidden layer MLP model obtained the worst accuracy (in orange). 
To visualize this better, the diagram below is the plot of the test accuracy score against the number of hidden units on this dataset for each 1, 2 and 5 hidden layers models:
![A1-5](https://user-images.githubusercontent.com/50171205/59688466-541f9a00-9210-11e9-9910-4211422d353d.png)

When the number of hidden units is small, the models lack the ability to learn with higher complexity to distinguish distinct differences among all 10 classes. However, as the number of hidden units increases, the model gains the ability to learn more complex representations. However, although the large number of hidden units may allow the training data to be fitted very well in the MLP model, a model with too many hidden units would most likely result in overfitting, where the model may fail to generalize to any new observations / data.

#### 2.1.2 Influence of Number of Hidden Layers on Test Accuracy 
Similarly, based on the visualization above, we can see that in general, MLP models that have more hidden layers tend to perform better. With a greater number of layers, the model increases in its capacity to learn more complex representations and thus potentially allow for better performance.

## Objective 2b
### 3.0 Summary of Results Obtained

| | Test Accuracy | Test Loss |
| --- | --- | --- |
| SGD Momentum = 0.9 | 0.886857 | 0.373575 |
| SGD Nesterov Momentum | 0.874686 | 0.409849 |
| Adagrad | 0.891886 | 0.494943 |
| RMSProp | 0.535886 | 1.803641 |
| ADAM	| 0.892571	| 0.372458 |

![A1-6](https://user-images.githubusercontent.com/50171205/59688468-5550c700-9210-11e9-8561-ad68f2217cc4.png)

### 3.1 Discussion On Results Obtained 
Based on the results in the table and plot above, we can deduce that ADAM has the highest Test Accuracy, while RMSProp has the lowest Test Accuracy. Conversely, RMSProp has the highest Test Loss while ADAM has the lowest Test Loss. However, this is based on the change in only one single parameter – the optimizer. There are other parameters such as learning rate that can be tuned to achieve better results. 
In this case, we observe that RMSProp has a substantially large amount of Test Loss. By taking a closer look at the Training Loss, we can see that it is substantially high as well, which could be an indication of exploding gradients. In order to address this, we can redesign the network to have fewer layers. By redesigning the model to have 2 hidden layers with 256 hidden units, we are able to achieve the following results:

| | Test Accuracy | Test Loss |
| --- | --- | --- |
| RMSProp	| 0.859371	| 0.654416 |

While we have eliminated that exploding gradients issue, the new test accuracy is still performing poorly as compared to the other optimizers (without tuning other parameters). 

## Objective 2c
### 4.0 Summary of Best MLP Model Results

| Model Name	| Epochs	| Batch Size	| Hidden Layers*	| Hidden Units	| Optimizer	| Test Accuracy Score	| Test Loss Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Adagrad_2c-2	| 30	| 16 | 5	| 16	| Adagrad	| 0.8918	| 0.3251 |

*Note: Within the five hidden layers, Batch Normalization and Dropout were used.*

### 4.1 Discussion on How This Configuration Was Reached
***Final Configuration for First Attempt: 50 Epochs | 16 Batch Size | 5 Hidden Layers | 512 Hidden Units | ADAM Optimizer With Batch Normalization & Dropout – “ADAM_2c-1”***
Based on the results obtained from question 2b, ADAM was the best optimizer with the following parameters: Epochs = 30, batch size = 16, hidden layers = 5 and hidden units = 256. Therefore, we started from that model and implemented changes along the way. In question 2b, the ADAM optimizer had a moderately high test accuracy rate, but has a test loss rate that is higher than that of its train loss rate. This may indicate overfitting, and thus we implemented regularization via dropout and batch normalization in the following model. Additionally, given that the model performance tend to increase (to a certain extent) with increasing number of hidden units, as shown in question 2a, we will also increase the number of our hidden units in the following model to 512.
The following table compares our best model from question 2b (“ADAM 2b”) to our first attempt (“ADAM 2c-1”) at any configuration:

| | Test Accuracy	| Test Loss |
| --- | --- | --- |
| ADAM_2b	| 0.8926	| 0.3725 |
| ADAM_2c-1	| 0.8775	| 0.3251 |

In our first attempt, our model ADAM_2c-1 did not perform as well as our first ADAM model from question 2b in Test Accuracy. However, the Test Loss in ADAM_2c-1 is significantly lower than that of ADAM_2b.
***Final Configuration for Second Attempt: 30 Epochs | 16 Batch Size | 5 Hidden Layers | 512 Hidden Units | Adagrad Optimizer with Batch Normalization & Dropout – “Adagrad_2c-2”***

On the second attempt,  another model was built based on another optimizer, the Adagrad optimizer, since it was the second best performing optimizer in question 2b. All other configurations were left unchanged except for the number of epochs, which was changed back to 30.
The following table compares all the models built thus far:

| |	Test Accuracy	| Test Loss |
| --- | --- | --- |
| ADAM_2b	| 0.8926	| 0.3725 |
| ADAM_2c-1	| 0.8775	| 0.3251 |
| Adagrad_2c-2	| 0.8919	| 0.3251 |

The second attempt fared much better such that we were able to obtain a significant improvement in Test Accuracy while Test Loss remains the same. Therefore, our final chosen model is the Adagrad_2c-2.
