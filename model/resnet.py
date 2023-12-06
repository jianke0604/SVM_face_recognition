import torch
import torch.nn as nn
import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, training=False):
        super(ResNet, self).__init__()
        if training:
            resnet = models.resnet18(weights="IMAGENET1K_V1")
        else:
            resnet = models.resnet18()
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if training:
            resnet.fc = nn.Linear(in_features=512, out_features=7, bias=True)
        else:
            resnet.fc = nn.Identity()
        self.backbone = resnet

    def forward(self, x):
        return self.backbone(x)
