from torch import nn


def linear_relu(dim_in, dim_out):
    return [nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True)]

class ClassifierModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ClassifierModel, self).__init__()
        self.classifier = nn.Sequential(
            *linear_relu(input_size, 256),
            *linear_relu(256, 256),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x_flatten = x.view(x.size()[0], -1)
        y = self.classifier(x_flatten)
        return y


