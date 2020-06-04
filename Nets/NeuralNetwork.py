from __future__ import absolute_import, print_function, division

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.utils.data as data_utils

class NeuralNetwork(object):

    def __init__(self, logger, data_dir, input_shape, num_classes, device='cpu'):
        self.model = None
        self.input_shape = input_shape
        self.data_dir = data_dir
        self.logger = logger
        self.num_classes = num_classes

        self.model_file_format = None
        self.model_files = []

        self.loss = None
        self.optimizer = None
        self.device = device

        self.train_data = None
        self.test_data = None
        self.test_loader = None

        self.norm_mu = None
        self.norm_sigma = None

        self.round = 0

        self.batch_size = 32

    def create_dataloader(self, subset, batch_size=None, shuffle=True):
        # create data loader with the above subset
        features = torch.stack([self.train_data[i][0] for i in subset])
        targets = torch.LongTensor([self.train_data[i][1] for i in subset])

        data = data_utils.TensorDataset(features, targets)
        if (batch_size == None):
            loader = data_utils.DataLoader(data, shuffle=True, batch_size=self.batch_size)
        else:
            loader = data_utils.DataLoader(data, shuffle=shuffle, batch_size=batch_size)
        return loader

    def train(self, subset, epochs, optimizer, test_set, learning_rate=0.1, momentum=0.9, batch_size=50, always_test=1):
        test_acc = None
        self.logger.info('Training started', extra={'props':
            {
                'lr': float(learning_rate),
                'momentum': float(momentum),
                'batch_size': batch_size,
                'data_size': len(subset)
            }
        })
        self.build_model()
        self._initialize_weights()
        self.set_hyperparams(optimizer, learning_rate, momentum, batch_size)
        self.model.to(self.device)
        self.model.train()
        train_loader = self.create_dataloader(subset) 
        train_loss = []
        test_loss = []   

        if(test_set != 'TEST'):
            self.test_loader = self.validate_loader

        for epoch in range(1, epochs + 1):
            loss_value = 0
            for batch_idx, (data, target) in enumerate(train_loader, 1):
                self.model.train()
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss(output, target)
                loss.backward()
                loss_value += loss.item()
                self.optimizer.step()
                self.logger.info('Training', extra={'props':
                    {
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'progress': '[{}/{} ({:.0f}%)]'.format(
                            batch_idx * len(data), len(subset),
                            ((100.0 * batch_idx * len(data)) / len(subset))
                        ),
                        'loss': float(loss)
                    }
                })
                print('train epoch: {} [{}/{} ({:.2f}%)]    loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(subset),
                    ((100. * batch_idx * len(data)) / len(subset)), loss.item()))
            train_loss.append(loss_value/len(train_loader))
            if (always_test):
                test_acc, l = self.test(epoch=epoch)               
            else:
                test_acc = 0
                l = 0
            test_loss.append(l)
            self.model.train()

        self.logger.info('Trained!', extra={'props': {'acc': test_acc}})
        print('Model trained!')
        return test_acc, train_loss, test_loss

    def test(self, **kwargs):
        self.model.eval()
        test_loss = 0
        correct = 0
        total_count = len(self.test_loader)
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.loss(output, target).item() # sum up batch loss
                pred = output.max(1, keepdim=True)[1] # get the index of the max log_probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        self.logger.info('Testing', extra={'props':
            {
                'avg_loss': float(test_loss),
                'acc': float(correct / len(self.test_loader.dataset)),
                'correct': correct,
                'train_epoch': kwargs.get('epoch', None)
            }
        })
        acc = (100. * correct) / len(self.test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(self.test_loader.dataset),
                ((100. * correct) / len(self.test_loader.dataset))))
        return acc, test_loss
    

    ### abstract methods ###
    def _initialize_weights(self):
        pass
    def build_model(self):
        raise NotImplementedError("Please use a concrete NeuralNetwork class!")

    def set_hyperparams(self, learning_rate, momentum, batch_size):
        raise NotImplementedError("Please use a concrete NeuralNetwork class!")

    def read_dataset(self):
        raise NotImplementedError("Please use a concrete NeuralNetwork class!")



