import numpy as  np 
import matplotlib.pyplot as plt
import train 
import  torch 
from  Perceptron_Model import Model


#--------------------------------
# initialize the model
M_Model = Model(2,1)
#  load the state dictionary
checkpoint = torch.load("Model1")
M_Model.load_state_dict(checkpoint["model_state_dict"])


#  set the model to evaluation mode 
M_Model.eval
#  get the paramters 
v = M_Model.get_param()

#get data 
data = checkpoint["Data"]["data1"]


#-------------------
def plot_fit(titile):
    plt.title = titile
    w1, w2, b1 = v
    x1 = np.array([-2, 2])
    x2 = (w1*x1 + b1)/(-w2)
    plt.plot(x1, x2, 'r')
    plt_pts()
    plt.show()

def plt_pts():
    x_data , y , _ = data
    plt.scatter(x_data[y == 0,0],x_data[y == 0,1])
    plt.scatter(x_data[y == 1,0],x_data[y == 1,1])
    plt.show()

plot_fit("test1")
