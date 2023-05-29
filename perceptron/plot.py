import numpy as  np 
import matplotlib.pyplot as plt
import train 
import Perceptron_Model

x_data , y , _ = train.Generate()

def plt_pts(x_data):
    plt.scatter(x_data[y == 0,0],x_data[y == 0,1])
    plt.scatter(x_data[y == 1,0],x_data[y == 1,1])
    plt.show()

#plt_pts(x_data)
