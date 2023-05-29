import numpy  as np  
import torch  
import torch.nn as  nn 
import matplotlib.pyplot as  plt


class  Model(nn.Module): 
    def __init__(self, input_size, output_size): 
        super().__init__()
        self.linear = nn.Linear(input_size,output_size)

    def forward(self, in_data):
        pred = torch.sigmoid(self.linear(in_data))
        return pred
    

