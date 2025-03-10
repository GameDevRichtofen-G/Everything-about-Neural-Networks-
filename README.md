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

### Activation function 

a mathematical function that can be applied to result of a neuron, we mostly use activation functions to determine whether neuron gonna get fire or not
or sometimes do other stuff like normalizing.

#### Relu 
ReLU or Rectified Linear Unit is an activation function which make neuron deactivate if the value of that neuron is 0 or less.
<div align="center">
  <img src="https://github.com/GameDevRichtofen-G/Everything-about-Neural-Networks-/blob/main/Relu_function.png"></br>
  <img src="https://latex.codecogs.com/svg.image?\large&space;{\color{white}Relu(x)=max(0,x)}">
  
  
</div>

#### Sigmoid
Sigmoid takes input and passes output between 0 and 1

<div align="center">
  <img src="https://github.com/GameDevRichtofen-G/Everything-about-Neural-Networks-/blob/main/Sigmoid_function.png"></br>
  <img src="https://latex.codecogs.com/svg.image?\large&space;{\color{white}Sigmoid(x)=\frac{1}{1&plus;e^{-x}}}">
</div>

#### SoftMax
Unlike sigmoid which compress single input, softmax compress multiple input.

<div align="center">
  <img src="https://github.com/GameDevRichtofen-G/Everything-about-Neural-Networks-/blob/main/Soft_max_function.png"></br>
  <img src="https://latex.codecogs.com/svg.image?\large&space;{\color{white}SoftMax(x_{i})=\frac{e^{x_{i}}}{\sum_{j}e^{x_{j}}}}">
</div>

we have more activation function to check but for now these 3 are the most common ones.









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


#### Getting the gradient
the `gradient` tells us the direction and rate of change of a loss function. which we can use this formula to get it : </br>
<div align="center">
  <img src="https://latex.codecogs.com/svg.image?{\color{white}\frac{\partial&space;L}{\partial&space;w}}">
</div>



### Optimizer 

optimizer is a operation that will adjust weight and bias to reduce the amount of the `loss` for certain prediction.
The most common optimizers 
#### Stochastic Gradient Descent (SGD)
SGD updates the weights in the opposite direction of the `gradient`:
<div align="center">
  <img src="https://latex.codecogs.com/svg.image?\large&space;{\color{white}Wnew=Wold-{\color{Yellow}a}\frac{\partial&space;L}{\partial&space;w}}">
</div>

![Description](https://latex.codecogs.com/svg.image?{\color{Yellow}a}) : Learning rate(it is a really small value use to tweak the adjustment by a little)

![Description](https://latex.codecogs.com/svg.image?\tiny&space;{\color{White}\frac{\partial&space;L}{\partial&space;W}}) : Gradient of the loss with respect to weight 


#### Adam (Adaptive Moment Estimation)
Adam is one of the most widely used optimizers and adapts the learning rate for each weight:

<div align="center">
  <img src="https://latex.codecogs.com/svg.image?\large&space;{\color{white}m_{t}=\beta_{1}m_{t-1}&plus;(1-\beta_{1}\frac{\partial&space;L}{\partial&space;W})}"> </br>
  <img src="https://latex.codecogs.com/svg.image?\large&space;{\color{white}v_{t}=\beta_{2}v_{t-1}&plus;(1-\beta_{2}(\frac{\partial&space;L}{\partial&space;W})^{2})}"> </br>
  <img src="https://latex.codecogs.com/svg.image?\large&space;{\color{white}\hat{m}_{t}=\frac{m_{t}}{1-\beta&space;_{t}^{1}}}">, <img src="https://latex.codecogs.com/svg.image?\large&space;{\color{white}\hat{v}_{t}=\frac{v_{t}}{1-\beta&space;_{v}^{2}}}">  </br>
  <img src="https://latex.codecogs.com/svg.image?\large&space;{\color{white}Wnew=Wold-a\frac{\hat{m}_{t}}{\sqrt{\hat{v}_t}&plus;\epsilon}}"> </br>
  
</div>

<sub> ![Description](https://latex.codecogs.com/svg.image?\large&space;{\color{white}\beta_{1},\beta_{2}}) </sub> : decay rates (commonly 0.9, 0.999)

![Description](https://latex.codecogs.com/svg.image?{\color{Yellow}a}) : Learning rate(it is a really small value use to tweak the adjustment by a little)

![Description](https://latex.codecogs.com/svg.image?\tiny&space;{\color{White}\frac{\partial&space;L}{\partial&space;W}}) : Gradient of the loss with respect to weight 

![Description](https://latex.codecogs.com/svg.image?\large&space;{\color{white}\epsilon}) : it is really small value to make sure we are not dividing with 0

![Description]( https://latex.codecogs.com/svg.image?\large&space;{\color{white}m_{t}}) : First Moment - Mean of Gradients

![Description]( https://latex.codecogs.com/svg.image?\large&space;{\color{white}v_{t}}) : Second Moment - Mean of Squared Gradients </br>
there are many different other optimizers, but these two are the most common ones for `binary cross entropy ` and `Multi Cross entropy`.

# How to create a NN
there are so many different ways to create our own Neural network. We can make one from scratch or we can one using different `tools` like : 


  
TensorFlow             |  Pytorch
:-------------------------:|:-------------------------:
![Description](https://i.postimg.cc/fR6FByfX/Tensor-Flow-logo-svg.png)  |  ![Description](https://i.postimg.cc/3N4mS8B4/Py-Torch-logo-white-svg-1.png)


We gonna use both of them to make a simple binary classification model.


### TensorFlow

First we need dataset, for this project I like to make a binary classification model that predicts the gender of a person(0 female and 1 male)
install the dataset csv from this github repository. </br>
Install TensorFlow, Numpy, plot using this command :
```
pip install matplotlib numpy tensorflow pandas
#Tensorflow -> for making a NN
#Numpy -> for setting up our dataset
#Pandas -> for reading our dataset
#matplot -> showing the result of the training
```
and then make a python file and call it TrainModel.py or what ever name you want.

Import these libraries to your python file :

```python 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as keras # I used keras since I can't no longer access it via tensorflow, but in general we will use this one to make our nn model
import copy as copy
```

after importing lets read and save our data set in array using numpy and pandas :

``` python

DATA = pd.read_csv("example_dataset.csv") -> reading the dataset

```

then we need to split our dataset to the group of 3 : `Train`, `Validation`, `Test`
```python
Train, Validation , Test  = np.split(DATA.sample(frac=1),[int(0.6 * len(DATA)), int(0.8 * len(DATA))])
```
the reason is when we training the model, we don't want to pass everything to it. 
because it technically memorize everything and it would be a bad model for a new data.

now, we need to make our inputs, by inputs I mean our features which we call them X, and our output as 
which we can use them to compare prediction. we do this for all of our dataset `Train`, `Validation`, `Test`.
to do this we will use function blow to split our data set to x and y

``` python

def GET_XandY(data_frame,#->our dataset
y_labels,#our y or outputs, usually the last column,
x_labels = None,#->This is customizialbe if you just wanna train model only on one column,
oversample = False) :#->oversampling is technic to make our dataset equal, For instance if we have 200 male and 100 woman, we will make dataset 100 male and 100 woman
    data_frame = copy.deepcopy(data_frame) #->getting a copy from our dataset
    if x_labels is None:
        x = data_frame[[c for c in data_frame.columns if c!=y_labels]].values#->if we didn't have any custom x lable we just pass all the column except of the column that contains our outputs
    else :
        if len(x_labels) == 1 :
            x = data_frame[x_labels[0]].values.reshape(-1,1) #-> if our xlable was just one column we pass it but we will fix it dimension
        else :
            x = data_frame[x_labels].values #-> just pass the custom x lable 

    y = data_frame[y_labels].values.reshape(-1,1) #->pass ylabes to get our y dataset, Note that since it 1 Dim we need to adjust its dim


   
   

    data = np.hstack((x,np.reshape(y,(-1,1)))) #->stack x and y in data variable

    return data,x,y #->return our data, x, y

```

and to get our x and y for `Train`,`Validation`,`Test` like this :
``` python
Train,XTrain,YTrain = GET_XandY(Train,"Gender",DATA.columns[:-1]#->passing all columns except the last column which is our predications)
Validation,XValidation,YValidation = GET_XandY(Validation,"Gender",DATA.columns[:-1]#->passing all columns except the last column which is our predications )
Test,XTest,YTest = GET_XandY(Test,"Gender",DATA.columns[:-1]#->passing all columns except the last column which is our predications)
```

Now that we have all the dataset we wanted we can actually make our model.
first we make function to create us a NN :
``` python
def CreateTheModel(
Num_Inputs#->Our Num x features
,DropOut, #-> our num of dropouts, we use this to randomally remove node
LR, #->Learning Rate or a
Num_Nodes, # Number of hidden layer neuron
Xtrain) :  #->our x features
    
    Norml_ = keras.layers.Normalization(input_shape = (Num_Inputs,),axis=-1)#->this basically our input layer, but we normalize our x features
    
    Norml_.adapt(Xtrain) #->the layer learns the dataset’s statistics so it can later standardize new inputs
    
    NN_model = keras.Sequential([#->to make a nn model we use code keras.Sequential
        #to pass values from a layer to different layer we use keras.layers.dense

        Norml_,#->passing the input layer
        keras.layers.Dense(Num_Nodes, activation="relu"),#->passing the inputs to relu function and passing the output of relu to a hidden layer 
        keras.layers.Dropout(DropOut),#->deactivating some of the neurons
        keras.layers.Dense(Num_Nodes,activation="relu"),#->passing our hidden layer values to a relu function and that would pass those value to our lass hidden layer
        keras.layers.Dense(1,activation="sigmoid")#->pass the last hidden layer value to 1 single output which would use sigmoid function to make output between 0 and 1
        
    ])
   
    NN_model.compile(optimizer=keras.optimizers.Adam(LR),loss="binary_crossentropy")#->compile our model, which it gonna use adam optimizer and binary cross entropy to train itselfr
    return NN_model #->return the model

```
Now to create our model we just do this :
```python
Model = CreateTheModel(len(DATA.columns[1:]),0,0.1,16,XTrain)
```

Now it is time to train it, to train our model we make another function : 
```python
def Train_theModel(
Input_model,#->our model
Xtrain,#->our x feature train
Ytrain,#->our y real train
XValid,#->our x feature validation to test the model on new data
YValid,#->our y real validation
Epochs,#->number of training sessions, in other word train model more
Batch_size,#->our sample size or batchsize
Show = False) : #->if we want we make this true to show the result

    History = Input_model.fit(Xtrain,Ytrain,epochs=Epochs,verbose=int(Show),validation_data=(XValid,YValid),batch_size = Batch_size)#train the model and then save the loss history
    return History #->return our history
```
Now to train we use this code :
``` python
  History = Train_theModel(Model,XTrain,YTrain,XValidation,YValidation,100,32,True)
```


To show our history data we use matplot libary like this :
``` python
def plot_loss(history) :
    plt.plot(history.history['loss'], label = "loss")#->showing the data for training loss
    plt.plot(history.history['val_loss'],label = "val_loss")#->showing the data for validation loss
    plt.title('Model loss')
    plt.ylabel('Loss')#->the amout of loss
    plt.xlabel('Epoch')#->the amount iterations
    plt.legend()
    plt.grid(True)
    plt.show()

```


then we do this :
``` python
plot_loss(History)
```

and now finally we can test our model by giving it our test data set and make our model predict:
```python
prediction = Model.predict(XTest)#->using function predict we can predict the result of our test dataset


for s in range(len(prediction)):#->going through each predication
    print("Predicted Output:", round(prediction[s][0]))#-> we round the prediction result to get either 0 or 1
    print("Actual Output:", YTest[s][0]) #->showing the actual result



```

### Pytorch 
This time we gonna make the same model in pytorch,
First we need dataset, for this project I like to make a binary classification model that predicts the gender of a person(0 female and 1 male)
install the dataset csv from this github repository. </br>
Install Torch, Numpy, plot and scikit learn using this command :
```
pip install torch numpy pandas scikit-learn matplotlib
#->torch : to make our nn model
#->numpy : to setup our dataset
#->pandas : to read our dataset
#->sklearn : to split the dataset
#->matplotlib : to show our results
```

and then make a python file and call it TrainModel.py or what ever name you want.

Import these libraries to your python file :

```python
import torch.nn as nn
import numpy as np
import pandas as pd
import torch as torch#->we use this for nn model
import torch.optim as opt#-> we use this to make our optimizer
from sklearn.preprocessing import StandardScaler 
from torch.utils.data import TensorDataset#-> turning np to torch tensor
from torch.utils.data import DataLoader#-> for loading a tensor
from sklearn.model_selection import train_test_split as split_data#->we use this to split data
```
after importing lets read and save our data set in array using numpy and pandas :

``` python
DATA = pd.read_csv("planets_data.csv")#->reading the csv file and save it as array
Features = DATA.drop(["Gender"],axis=1)#->removing gender column from dataset and pass other columns to x features
Target = DATA["Gender"].values.reshape(-1,1)#->getting the result of x features
```
then we need to split our dataset to the group of 2 : `Train`, `Validation`

```python
Train_x,validation_x,Train_y,validation_y  = split_data(Features,Target,test_size=0.2,random_state=42)#->spliting dataset between x train, y train and x validation and y validation 
```

the reason is when we training the model, we don't want to pass everything to it. 
because it technically memorize everything and it would be a bad model for a new data.

we split our dataset to x and y where x are actually our features and our output as y
which we can use them to compare prediction. we did this for all of our dataset `Train`, `Validation`
</br>

Now we need to turn our numpy arrays to torch tensor using this code :

```python
X_train_tensor = torch.FloatTensor(Train_x.values)#->turning our train features to torch tensor
Y_train_tensor = torch.FloatTensor(Train_y)#->turning our train output  to torch tensor

X_validation_tensor = torch.FloatTensor(validation_x.values)#->turning our validation features to torch tensor
Y_validation_tensor = torch.FloatTensor(validation_y)#->turning our validation output to torch tensor
```

Now we need a general dataset which we can make using TensorDataset:

```python
Train_data_torch = TensorDataset(X_train_tensor,Y_train_tensor)#->making a general dataset for train
Validation_data_torch = TensorDataset(X_validation_tensor,Y_validation_tensor)#->making a general dataset for validation
```
Now we need to load data using a dataloader :
```python
Train_loaded = DataLoader(Train_data_torch,batch_size=32, shuffle= True)#->making train data_loader with shuffeling and  sample size of 32
Validation_loaded = DataLoader(Validation_data_torch,batch_size=32)#->making validation data_loader and  sample size of 32

```

now that we have our dataset ready lets make our NN.
to do that we to make class like this :
```python
class NerualNModel(nn.Module) :
.........
```

we need to setup our NN layers and activation function in Init function like this :
```python
    def __init__(self,
    num_inputs,#-> number of x features
    DropOut, #->chance of dropout
    Num_hidden, #->num of hidden layers
    Num_outputs): #->num of outputs
        super(NerualNModel,self).__init__()
        #to make a value get transfer first layer to another, we use nn.linear
        self.normal = nn.BatchNorm1d(num_inputs)#->creating a normalized layer for inputs. in other word make the value of Xn between 0 or 1
        
        self.Relu_function = nn.ReLU()#->creating a relu function
        
        self.DropOut = nn.Dropout(DropOut)#->#adding the dropout, basically deactivate a neuron randomally 
        
        self.layer1 = nn.Linear(num_inputs,Num_hidden)#->Input to first hidden layer 
        
        self.layer2 = nn.Linear(Num_hidden,Num_hidden)#->from first hidden layer to second hidden layer
        
        self.layer3 = nn.Linear(Num_hidden,Num_outputs)#->from second hidden layer to the output lays
        self.sigmoid = nn.Sigmoid()#->make our sigmoid function, to return a value between 0 and 1.
```

Now we are done with setup the nn lets make forward propagation, in same class :
```python
    def forward(self,x):
        x = self.normal(x)#-> passing our features to input layer which in our case it would normalize the values
        x = self.Relu_function(self.layer1(x))#->pass our inputs to relu function and then we pass the result to first hidden layer
        x = self.DropOut(x) #->randomally removing a neuron
        x = self.Relu_function(self.layer2(x)) #->pass our first hidden layer to relu function and then we pass the result to second hidden layer
        x = self.sigmoid(self.layer3(x))#->pass the value from last hidden layer to ouput layer and output layer will use sigmoid to make our value between 0 and 1
        return x
```

Alright we made our NN class lets make a model and train it,
first we need to get the number of x features, which we would do like this :
```python
Number_features = Train_x.shape[1]
```

to create our model and save it we do this :
```python
NNmodel = NerualNModel(num_inputs=Number_features,DropOut=0.2,Num_hidden=64,Num_outputs=1)
```
now we need to calculate the loss, which we can just use a loss function and we need an optimizer as well so to do that we do this :
```python

loss_function = nn.BCELoss()#->making a binary cross entropy loss function

optimizer = opt.Adam(NNmodel.parameters(),lr=0.001)#->setting up a adam optimizer
```

Now that is done we can do our training for certain amount of iteration like this :
```python
epochs = 100
for epoch in range(epochs):
    
    NNmodel.train()#->activating the training mode

    losses = []#->making an history of our losses


    for inputs, targets in Train_loaded:#->going thourgh our dataset
        optimizer.zero_grad() #->reseting the gradient
        outputs = NNmodel(inputs) #->redict the output base on the given inputs
        loss = loss_function(outputs,targets) #->getting the loss
        loss.backward() #->do backpropergation
        optimizer.step()#->update the NN
        losses.append(loss.item())#->save the  loss


    ave_train_loss = sum(losses) / len(losses)#-> getting the average loss

    #doing a test for validation dataset
    NNmodel.eval()#->evaluate our Model
    val_loss = []#->make history of our validation losses
    with torch.no_grad():#->No training
        for inputs, targets in Validation_loaded:#->going through all data in our valivation dataset
            output = NNmodel(inputs)#->getting an output
            loss = loss_function(output,targets)#->use our loss function to caluclate the loss
            val_loss.append(loss.item())#->save that loss

    ave_validation_loss = sum(val_loss) / len(val_loss)#-> calculate the average loss

    #debugging each iteration
    print(f'Epochs : {epoch+1}/{epochs}, TrainLoss : {ave_train_loss}, validationLoss : {ave_validation_loss}')

```

after training we can actually test the model :
```python
        sample_input = np.array([[163.8053817893671,54.73277335389531,243.03149704782862,21.499301492387644]])#->giving a test which should give output of 0
        NNmodel.eval()#->evaluate the model
        tesnor_input = torch.FloatTensor(sample_input)#->turn numpy array to tensor
        with torch.no_grad():#->no training
            predication = NNmodel(tesnor_input)#->getting our predication base on our tensor input
        Result =  round(predication.item())#->getting the result
```




