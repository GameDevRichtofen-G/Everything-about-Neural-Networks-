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
And each neuron has it own `Bias`, which can be defined as the constant which is added to the product of features and weights.</br>
To transfer values from first Neuron to second Neuron we use this equation :
</br>
<div align="center">
  <img src="https://latex.codecogs.com/svg.image?\LARGE&space;{\color{White}b&plus;\sum_{i=1}^{n}x_{i}w_{i}}">
</div>

the simplify version of this formula would be  <sub> ![Description](https://latex.codecogs.com/svg.image?{\color{White}Y=W_{i}\times&space;X_{i}&plus;bias}) </sub>
</br>by doing this we can transfer value from one neuron to another. This process is called  `Forward Propagation`
## Layers
ANNs are made out 3 general layer.<br/>
`INPUT` : Takes <sub> ![Description](https://latex.codecogs.com/svg.image?{\color{White}X_{1}\cdots&space;X_{n}}) </sub> as input</br>
`HIDDEN` : 

