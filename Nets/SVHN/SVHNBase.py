import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from Nets import NeuralNetwork, broadcast_to_shape

class SVHNBase(NeuralNetwork):

    def __init__(self, logger, data_dir, device='cpu'):

        super(SVHNBase, self).__init__(logger, data_dir, (3, 32, 32), 10)
        self.norm_mu = (0.4376821, 0.4437697, 0.47280442)
        self.norm_sigma = (0.19803012, 0.20101562, 0.19703614)

    def set_hyperparams(self, optimizer, learning_rate=0.1, momentum=0.9, batch_size=50):
        self.loss = nn.CrossEntropyLoss()
        if (optimizer == 'SGD'):
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
        elif(optimizer == 'ADAM'):
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif(optimizer == 'RMS'):
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate)
        self.batch_size = batch_size

    def read_dataset(self, batch_size=1):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(self.norm_mu, self.norm_sigma)])

        self.train_data = datasets.SVHN('./data', split='train', download=True, transform=transform)
        self.test_data = datasets.SVHN('./data', split='test', download=True, transform=transform)
        self.test_loader = torch.utils.data.DataLoader(self.test_data,shuffle=True)
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size)
        self.logger.info('Data Loaded')

    def build_model(self):
        raise NotImplementedError("Please use a concrete SVHNBase class!")

