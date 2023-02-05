import torch
from torch import nn
from torch.nn import Conv2d
from torch.nn import functional as F


class ResidualBlock(nn.Module):

    def __init__(self, input_dim, kernel_size):
        super().__init__()
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.conv = Conv2d(input_dim, input_dim, kernel_size, padding='same')

    def forward(self, x):
        x_init = torch.clone(x)
        x = self.conv(x)
        x = F.relu(x)
        x = self.conv(x)
        x = F.relu(x)
        return x_init + x


class SimpleResidualClassifier(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        # first convulotional block
        self.initial_conv = nn.Sequential(
            Conv2d(input_dim, 16, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # first residual block
        self.resblock1 = ResidualBlock(16, 3)

        # second convolutional block
        self.second_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            Conv2d(16, 64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
        )

        # second residual block
        self.resblock2 = ResidualBlock(64, 3)

        # final dense classifier
        self.final_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.resblock1(x)
        x = self.second_conv(x)
        x = self.resblock2(x)
        x = self.final_classifier(x)
        return x
