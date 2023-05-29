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

#initialize a Model with 2 inputs and 1 output features
torch.manual_seed(2)
M_Model = Model(2,1)


# train function 

def train_():
    x_data, _ , y_data = Generate()
    # set the learning rate 
    learning_rate = 0.5
    #  set the epochs
    epochs = 100
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
        print("epoch:",i ,"loss:", loss.item())
        # third calculate the gradient
        optimizer.zero_grad()
        loss.backward()
        # update  the parametrs
        optimizer.step()

train_()

# extract weight and bias

def get_parameters():
    return None