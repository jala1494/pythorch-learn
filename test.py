import torch
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import torch.nn as nn
input=torch.randn(128,20)
print(input)
m=nn.Linear(20,30)
print(m)
output=m(input)
print(output)
print(output.size())

input=torch.randn(20,16,50,100)
print(input.size())
m=nn.Conv2d(16,33,3,stride=2)

m=nn.Conv2d(16,33,(3,5),stride=(2,1),padding=(4,2))
m=nn.Conv2d(16,33,(3,5),stride=(2,1),padding=(4,2),dilation=(3,1))
print(m)/jala1494.github.io/index.html
output=m(input)
print(output.size())

nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
layer=nn.Conv2d(1,20,5,1).to(torch.device('cpu'))
print(layer)

weight=layer.weight
print(weight.shape)
weight=weight.detach()
weight=weight.numpy()
print(weight.shape)

plt.imshow(weight[0,0,:,:],'gray')
plt.colorbar()
plt.show()