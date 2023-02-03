import torch
from torch import nn
from torch.nn import Sequential, Conv2d, AvgPool2d, Linear, Sigmoid, Softmax, Flatten


class LeNet(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.model = Sequential(
            Conv2d(input_channels, 6, kernel_size=(5, 5), padding='same'),
            Sigmoid(),
            AvgPool2d(kernel_size=(2, 2)),
            Conv2d(6, 16, kernel_size=(5, 5), padding='valid'),
            Sigmoid(),
            AvgPool2d(kernel_size=(2, 2)),
            Conv2d(16, 120, kernel_size=(5, 5), padding='valid'),
            Sigmoid(),
            Flatten(),
            Linear(in_features=120, out_features=84),
            Sigmoid(),
            Linear(in_features=84, out_features=10),
            Softmax()
        )

    def forward(self, x):
        x = self.model(x)
        return x
