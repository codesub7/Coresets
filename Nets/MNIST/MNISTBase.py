import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from Nets import NeuralNetwork, broadcast_to_shape

class MNISTBase(NeuralNetwork):

    def __init__(self, logger, data_dir, device='cpu'):

        super(MNISTBase, self).__init__(logger, data_dir, (1, 28, 28), 10)
        self.norm_mu = (0.1307, )
        self.norm_sigma = (0.3081, )

    def set_hyperparams(self, optimizer, learning_rate=0.1, momentum=0.9, batch_size=50):
        self.loss = nn.CrossEntropyLoss()
        if (optimizer == 'SGD'):
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
        elif(optimizer == 'ADAM'):
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif(optimizer == 'RMS'):
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate)
        self.batch_size = batch_size

    def read_dataset(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(self.norm_mu, self.norm_sigma)])
        self.train_data = datasets.MNIST(root=self.data_dir, transform=transform,
            train=True, download=True)
        self.test_data = datasets.MNIST(root=self.data_dir, transform=transform,
            train=False, download=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_data,shuffle=True)
        self.train_loader = torch.utils.data.DataLoader(self.train_data)
        self.logger.info('Data Loaded')

    def transform_data_for_forward_pass(self, x):
        if isinstance(x, torch.Tensor):
            data = x.detach().numpy()
        else:
            data = x.copy()
        assert(len(data.shape) == 3 or len(data.shape) == 4)
#        if np.all(0.0 <= data) and np.all(data <= 1.0):
#            data = self.normalizeInput(data)
        if isinstance(data, np.ndarray):
            data = torch.tensor(data)
        if len(data.shape) == 3:
            data = data.unsqueeze(0)
        return data.float()

    def normalize_input(self, x):
        assert isinstance(x, np.ndarray)
        z = x.copy()
        z = z[0] if len(z.shape) == 4 else z
        for i in range(z.shape[0]):
            z[i] -= self.norm_mu[i]
            z[i] /= self.norm_sigma[i]
        return z[np.newaxis, :]

    def unnormalize_input(self, x):
        assert isinstance(x, np.ndarray)
        z = x.copy()
        z = z[0] if len(z.shape) == 4 else z
        for i in range(z.shape[0]):
            z[i] *= self.norm_sigma[i]
            z[i] += self.norm_mu[i]
        return z[np.newaxis, :]

    def predict(self, x):
        x = self.transform_data_for_forward_pass(x)
        output = self.model(x)
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        return pred.item()

    def build_model(self):
        raise NotImplementedError("Please use a concrete MNISTBase class!")

