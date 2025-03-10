import torch.nn as nn
import numpy as np
import pandas as pd
import torch as torch#->we use this for nn model
import torch.optim as opt#-> we use this to make our optimizer
from sklearn.preprocessing import StandardScaler 
from torch.utils.data import TensorDataset#-> turning np to torch tensor
from torch.utils.data import DataLoader#-> for loading a tensor
from sklearn.model_selection import train_test_split as split_data#->we use this to split data

DATA = pd.read_csv("example_data_set.csv")#->reading the csv file and save it as array
Features = DATA.drop(["Gender"],axis=1)#->removing gender column from dataset and pass other columns to x features
Target = DATA["Gender"].values.reshape(-1,1)#->getting the result of x features
Train_x,validation_x,Train_y,validation_y  = split_data(Features,Target,test_size=0.2,random_state=42)#->spliting dataset between x train, y train and x validation and y validation 

X_train_tensor = torch.FloatTensor(Train_x.values)#->turning our train features to torch tensor
Y_train_tensor = torch.FloatTensor(Train_y)#->turning our train output  to torch tensor

X_validation_tensor = torch.FloatTensor(validation_x.values)#->turning our validation features to torch tensor
Y_validation_tensor = torch.FloatTensor(validation_y)#->turning our validation output to torch tensor

Train_data_torch = TensorDataset(X_train_tensor,Y_train_tensor)#->making a general dataset for train
Validation_data_torch = TensorDataset(X_validation_tensor,Y_validation_tensor)#->making a general dataset for validation

Train_loaded = DataLoader(Train_data_torch,batch_size=32, shuffle= True)#->making train data_loader with shuffeling and  sample size of 32
Validation_loaded = DataLoader(Validation_data_torch,batch_size=32)#->making validation data_loader and  sample size of 32

class NeuonNmodel(nn.Module) :
        def __init__(self,
            num_inputs,#-> number of x features
            DropOut, #->chance of dropout
            Num_hidden, #->num of hidden layers
            Num_outputs): #->num of outputs
             super(NeuonNmodel,self).__init__()
                #to make a value get transfer first layer to another, we use nn.linear
             self.normal = nn.BatchNorm1d(num_inputs)#->creating a normalized layer for inputs. in other word make the value of Xn between 0 or 1
        
             self.Relu_function = nn.ReLU()#->creating a relu function
        
             self.DropOut = nn.Dropout(DropOut)#->#adding the dropout, basically deactivate a neuron randomally 
        
             self.layer1 = nn.Linear(num_inputs,Num_hidden)#->Input to first hidden layer 
        
             self.layer2 = nn.Linear(Num_hidden,Num_hidden)#->from first hidden layer to second hidden layer
        
             self.layer3 = nn.Linear(Num_hidden,Num_outputs)#->from second hidden layer to the output lays
             self.sigmoid = nn.Sigmoid()#->make our sigmoid function, to return a value between 0 and 1.
            

        def forward(self,x):
             x = self.normal(x)#-> passing our features to input layer which in our case it would normalize the values
             x = self.Relu_function(self.layer1(x))#->pass our inputs to relu function and then we pass the result to first hidden layer
             x = self.DropOut(x) #->randomally removing a neuron
             x = self.Relu_function(self.layer2(x)) #->pass our first hidden layer to relu function and then we pass the result to second hidden layer
             x = self.sigmoid(self.layer3(x))#->pass the value from last hidden layer to ouput layer and output layer will use sigmoid to make our value between 0 and 1
             return x
        

Number_features = Train_x.shape[1]

NNmodel = NeuonNmodel(num_inputs=Number_features,DropOut=0.2,Num_hidden=64,Num_outputs=1)


loss_function = nn.BCELoss()#->making a binary cross entropy loss function

optimizer = opt.Adam(NNmodel.parameters(),lr=0.001)#->setting up a adam optimizer

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

sample_input = np.array([[163.8053817893671,54.73277335389531,243.03149704782862,21.499301492387644]])#->giving a test 
NNmodel.eval()#->evaluate the model
tesnor_input = torch.FloatTensor(sample_input)#->turn numpy array to tensor
with torch.no_grad():#->no training
    predication = NNmodel(tesnor_input)#->getting our predication base on our tensor input
    Result =  round(predication.item())#->getting the result

print(Result)

