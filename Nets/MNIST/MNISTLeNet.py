import torch
import torch.nn as nn

from Nets import Flatten
from .MNISTBase import MNISTBase

class MNISTLeNet(MNISTBase):
    def __init__(self, logger, data_dir, device='cpu'):
        super(MNISTLeNet, self).__init__(logger, data_dir, device)

    def build_model(self):
        self.model = nn.Sequential(
                        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        Flatten(),
                        nn.Linear(16*5*5, 120),
                        nn.ReLU(),
                        nn.Linear(120, 84),
                        nn.ReLU(),
                        nn.Linear(84, 10)
                  )

