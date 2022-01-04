import torch
import torchvision
import torch.nn as nn

class ResNetRegressor(nn.Module):
    """
    Build a regression model by calling a constructor in torchvision.models subpackage.
    """
    def __init__(self, out_channels, pretrained=True):
        super(ResNetRegressor, self).__init__()

        self.resnet = torchvision.models.resnet50(pretrained=pretrained)
        
        self.fc = nn.Linear(1000, out_channels)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        sig = nn.Sigmoid()
        return sig(x)