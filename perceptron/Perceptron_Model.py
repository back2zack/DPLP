import numpy  as np  
import torch  
import torch.nn as  nn 
import matplotlib.pyplot as  plt


class  Model(nn.Module): 
    def __init__(self, input_size, output_size): 
        super().__init__()
        self.linear = nn.Linear(input_size,output_size)
        self.data = dict()
        self.learning_rate = 0.4
        self.epochs = 100

    def forward(self, in_data):
        pred = torch.sigmoid(self.linear(in_data))
        return pred
    def get_param(self):
        [W, B] = self.parameters()
        W1 ,W2 = W.view(2)
        return (W1.item(), W2.item(), B[0].item())
    
    def genreated_data(self, Data :tuple ,key :str):
        self.data[key] = Data
        return self.data
