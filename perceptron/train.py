import numpy  as np  
import torch  
import torch.nn as  nn 
from Perceptron_Model import Model
from sklearn import  datasets





# prepare data  
def Generate():
    num_pts = 100
    centers = [[-0.5,0.5],[0.5,-0.5]]
    X ,y = datasets.make_blobs(n_samples=num_pts , random_state=123 , centers = centers , cluster_std= 0.4)

    # convert to tensors # shoudl all be float32 
    points  = torch.tensor(X)
    points = points.to(torch.float32)
    labels = torch.tensor(y)
    labels = labels.reshape(100,1)
    labels = labels.float()
    
    return points, y, labels


# train function 
def train_(Model:Model, data_key :str):
    data_dict = Model.data
    x_data, _ , y_data = data_dict[data_key]
    # set the learning rate 
    learning_rate = Model.learning_rate
    #  set the epochs
    epochs = Model.epochs
    # set the loss function
    criterion = nn.BCELoss()
    # optimizer
    optimizer = torch.optim.SGD(M_Model.parameters() , lr = learning_rate)
    # list of losses  to observe the learning process
    Losses = []
    # for  loop to optimaize the parametrs
    for i in range(epochs): 
        ## first make a predection
        y_pred = M_Model.forward(x_data)
        #print("y_pred",type(y_pred), "Y_data", type(y_data))
        # calculate  loss
        loss = criterion(y_pred, y_data)
        Losses.append(loss.item())
        #print("epoch:",i ,"loss:", loss.item())
        # third calculate the gradient
        optimizer.zero_grad()
        loss.backward()
        # update  the parametrs
        optimizer.step()



#initialize a Model with 2 inputs and 1 output features
torch.manual_seed(2)
M_Model = Model(2,1)

# generate  the training data for  our model
Data = Generate() 

# append that data to the model class-> 
# because if we generate new data poitns
# we need to retrain the model for the new data points 
M_Model.genreated_data(Data,"data1")


# train the model
train_(M_Model, "data1")

# Save the model's state dictionary to use these parameters in the plot file 
torch.save( {
    'model_state_dict' :M_Model.state_dict(),
    'Data': M_Model.data,
    },"Model1")
