import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as keras # I used keras since I can't no longer access it via tensorflow, but in general we will use this one to make our nn model
import copy as copy



def GET_XandY(data_frame,#->our dataset
y_labels,#our y or outputs, usually the last column
x_labels = None,#->This is customizialbe if you just wanna train model only on one column
oversample = False):#->oversampling is technic to make our dataset equal, For instance if we have 200 male and 100 woman, we will make dataset 100 male and 100 woman) :
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


def CreateTheModel(
Num_Inputs#->Our Num x features
,DropOut, #-> our num of dropouts, we use this to randomally remove node
LR, #->Learning Rate or a
Num_Nodes, # Number of hidden layer neuron
Xtrain): #->our x features)
    
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

def Train_theModel(
Input_model,#->our model
Xtrain,#->our x feature train
Ytrain,#->our y real train
XValid,#->our x feature validation to test the model on new data
YValid,#->our y real validation
Epochs,#->number of training sessions, in other word train model more
Batch_size,#->our sample size or batchsize
Show = False#->if we want we make this true to show the result
) :
    History = Input_model.fit(Xtrain,Ytrain,epochs=Epochs,verbose=int(Show),validation_data=(XValid,YValid),batch_size = Batch_size)#->whether to show process or not, train the model and then save the loss history
    return History #->return our history


def plot_loss(history) :
    plt.plot(history.history['loss'], label = "loss")#->showing the data for training loss
    plt.plot(history.history['val_loss'],label = "val_loss")#->showing the data for validation loss
    plt.title('Model loss')
    plt.ylabel('Loss')#->the amout of loss
    plt.xlabel('Epoch')#->the amount iterations
    plt.legend()
    plt.grid(True)
    plt.show()

DATA = pd.read_csv("example_data_set.csv")


Train, Validation , Test  = np.split(DATA.sample(frac=1),[int(0.6 * len(DATA)), int(0.8 * len(DATA))])
Train,XTrain,YTrain = GET_XandY(Train,"Gender",DATA.columns[:-1])#->passing all columns except the last column which is our predications)
Validation,XValidation,YValidation = GET_XandY(Validation,"Gender",DATA.columns[:-1])#->passing all columns except the last column which is our predications )
Test,XTest,YTest = GET_XandY(Test,"Gender",DATA.columns[:-1])#->passing all columns except the last column which is our predications)


Model = CreateTheModel(len(DATA.columns[1:]),0,0.1,16,XTrain)
History = Train_theModel(Model,XTrain,YTrain,XValidation,YValidation,100,32,True)
valdLoss = Model.evaluate(XValidation, YValidation)

plot_loss(History)

prediction = Model.predict(XTest)#->using function predict we can predict the result of our test dataset


for s in range(len(prediction)):#->going through each predication
    print("Predicted Output:", round(prediction[s][0]))#-> we round the prediction result to get either 0 or 1
    print("Actual Output:", YTest[s][0]) #->showing the actual result


