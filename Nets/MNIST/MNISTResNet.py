import torch
import torch.nn as nn

from torchvision.models.resnet import ResNet, BasicBlock

from Nets import Flatten
from .MNISTBase import MNISTBase

class MNISTResNet(MNISTBase):
    def __init__(self, logger, data_dir, device='cpu'):
        super(MNISTResNet, self).__init__(logger, data_dir, device)

    def build_model(self):
        self.model = ResNet(BasicBlock, [2,2,2,2], num_classes=10)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
