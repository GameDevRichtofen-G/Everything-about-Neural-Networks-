# What is Artificial Neural Network
Artificial Neural Networks are one the most powerful ML/DL models, capable of solving complex problems in simliar ways
like us humans. NN was found in 1958, where psychologist 		**Frank Rosenblatt**. He was the first person
who manage to implement the first NN model. These images shows the simliarity between humans NN and ML/DL ANN :
</br>
</br>
Human NN             |  AI ANN
:-------------------------:|:-------------------------:
![Description](https://github.com/GameDevRichtofen-G/Everything-about-Neural-Networks-/blob/main/Neuron6.png)  |  ![Description](https://github.com/GameDevRichtofen-G/Everything-about-Neural-Networks-/blob/main/Neuron5.png)




# Structure
ANNs have clear and genius structure. In general they are made out multiple layers and nodes which we will talk about in the moment. 

## Neurons
an ANN consists of connected units or nodes called neurons. Each neuron/node receive inputs from connected neurons,
and then the same neuron sends an input to other connected neurons:
<div align="center">
  <img src="https://i.postimg.cc/L6WJk4rR/Neuron7.png">
</div>


Each connection has its own `Weight`, which determines how likely it is for that individual neuron to be activated.</BR>
And each neuron has it own `Bias`, which can be defined as the constant which is added to the product of features and weights.
## Layers
### ANNs are made out 3 general layer.<br/>
`INPUT` : Takes <sub> ![Description](https://latex.codecogs.com/svg.image?{\color{White}X_{1}\cdots&space;X_{n}}) </sub> as input</br>
`HIDDEN` : the layers of neurons that are situated between the `INPUT` layer and the `OUTPUT` layer</br>
`OUTPUT` : The last layer of NN model that gives out <sub> ![Description](https://latex.codecogs.com/svg.image?\large&space;{\color{White}y_{n}}) </sub></BR>
<div align="center">
  <img src="https://i.postimg.cc/X7Msx5t7/Neuron8-1.png">
</div>


# Functionality
ANN has a pefect algorithm that allows it to trains itself well and solve complex problems
We will talk about its algorithm in a moment

## Forward Propagation 

As mention earlier each connection to a neuron has `Weight` and each Neuron has something called `Bias`.
we will use these parameters to transfer values with an adjustment.
To transfer values from first Neuron to second Neuron we use this equation :
</br>
<div align="center">
  <img src="https://latex.codecogs.com/svg.image?\LARGE&space;{\color{White}b&plus;\sum_{i=1}^{n}x_{i}w_{i}}">
</div>

the simplify version of this formula would be  <sub> ![Description](https://latex.codecogs.com/svg.image?{\color{White}Y=W_{i}\times&space;X_{i}&plus;bias}) </sub>
</br>by doing this we can transfer value from one neuron to another. This process is called  `Forward Propagation`

## Backward Propagation
`Backward Propagation` is just like `Forward Propagation` but we start from output layer.
we use `Backward Propagation` to train our model. This process is done via few steps :
### Calculating the Loss
Loss is the difference between the predicted output and the actual target value.
the equation to calculating the loss can be different base on the type of a model
#### binary cross entropy : 
the use case for this loss function is for models that have only 2 ouput value, either 0 or 1
</br>
<div align="center">
  <img src="https://latex.codecogs.com/svg.image?\LARGE&space;{\color{white}L=-\frac{1}{N}\sum_{N}^{i=1}[y_{real}log(y_{pred}&plus;(1-y_{real})log(1-y_{pred}))]}">
</div>
N = Number of samples (batch size)</br> </br> 

![Description](https://latex.codecogs.com/svg.image?{\color{white}y_{real}}) the actual value

![Description](https://latex.codecogs.com/svg.image?{\color{white}y_{pred}}) the predicted value
#### Multi Cross entropy
this loss function is useful if you have model that needs to predict a value more than 0 or 1

<div align="center">
  <img src="https://latex.codecogs.com/svg.image?\LARGE&space;{\color{white}L=-\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{C}Yreal_{i},_{k}log(Ypred_{i},_{k})}">
</div>
N = Number of samples (batch size)</br> </br> 

C = Number of classes</br> </br> 

![Description](https://latex.codecogs.com/svg.image?{\color{white}Yreal_{i},_{k}}) = One-hot encoded actual class label (1 if sample i belongs to class k otherwise 0)

![Description](https://latex.codecogs.com/svg.image?{\color{white}Ypred_{i},_{k}}) = Predicted probability for class k
