import torch.nn as nn
import torch
from torch import tensor
from typing import Any

##predefines what should be imported whilst using from model import *
__all__ = ["AlexNet", "alexnet"]

# Underlying architecture of the model
# It consists of 8 layers, 5 of which are conv layers and 3 are dense layers
# A softmax layer in the last layer for classifying into 1000 classes

class AlexNet(nn.Module):
    "nn models consisting of layers proposed by AlexNet"
    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()

        # input size should be : (k x 3 x 227 x 227)
        # The image in the original paper states that width and height are 224 pixels, but
        # the dimensions after first convolution layer do not lead to 55 x 55.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # (k x 96 x 55 x 55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (k x 96 x 27 x 27)
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),  # (k x 256 x 27 x 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (k x 256 x 13 x 13)
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),  # (k x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),  # (k x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),  # (k x 256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (k x 256 x 6 x 6)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

        self.init_bias()

    def init_bias(self):
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        
        # Biases initialization based on the paper
        nn.init.constant_(self.features[4].bias, 1)
        nn.init.constant_(self.features[7].bias, 1)
        nn.init.constant_(self.features[9].bias, 1)

    def forward(self, x):
        """Pass the input through the network
        
        Args:
            x (tensor): input tensor
            
        Returns:
            output (tensor): output tensor
        """
        x = self.features(x)
        x = x.view(-1, 256 * 6 * 6)  # Flattening
        x = self.classifier(x)
        return x

def alexnet(**kwargs: Any) -> AlexNet:
    model = AlexNet(**kwargs)
    return model
