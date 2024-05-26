#Pytorch

I am learning pytorch. I want to apply this in 1&1 project.

## Installation
conda install pytorch torchvision torchaudio cpuonly -c pytorch
https://pytorch.org/get-started/locally/

### Installtion check

import torch
x = torch.rand(5, 3)
print(x)

## Data Handling

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
