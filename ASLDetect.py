import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.models import resnet34, ResNet34_Weights
import torchvision
import torch.nn.functional as F
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.ops import SqueezeExcitation



# ResNetUNet Model Definition
class ResNetUNet(nn.Module):
    def __init__(self, n_classes):
        super(ResNetUNet, self).__init__()

        self.base_model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        self.base_layers = list(self.base_model.children())
        
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.conv_up3 = self.conv_block(512, 256)
        self.conv_up2 = self.conv_block(256, 128)
        self.conv_up1 = self.conv_block(128, 64)

        self.conv_last = nn.Conv2d(128, 64, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, n_classes)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.upconv4(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up3(x)

        x = self.upconv3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)

        x = self.upconv2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up1(x)

        x = self.upconv1(x)
        x = torch.cat([x, x0], dim=1)
        x = self.conv_last(x)

        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        output = self.classifier(x)
        return output
    
