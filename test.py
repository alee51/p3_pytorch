#import torch
#
#data = [[1, 2],[3, 4]]
#x_data = torch.tensor(data)
#
#x_ones = torch.ones_like(x_data) # retains the properties of x_data
#print(f"Ones Tensor: \n {x_ones} \n")
#
#x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
#print(f"Random Tensor: \n {x_rand} \n")

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
