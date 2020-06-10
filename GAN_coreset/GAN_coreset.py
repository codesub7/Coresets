import sys
sys.path.append('../')
import logging
import torch
import numpy as np
import argparse

from Nets.MNIST import MNISTResNet

from util import *
import torch.nn.functional as F
import time
import pickle

def main(args):

    logger = logging.getLogger("Active-Learning-NN-Train")
    logger.setLevel((6 - args.verbose) * 10)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    net = MNISTResNet(logger, './data')

    net.read_dataset()
    net.build_model()
    logger.info('Completed reading dataset; built NN model')

    #load the latent space mappings of mnist data

    with open('../mnist-latentspace.pickle','rb') as f:
        features = pickle.load(f)['z']

    #choose a random pool of data
    random_pool = np.arange(len(net.train_data))
    np.random.shuffle(random_pool)
    random_pool = random_pool[:args.init_budget]
    curr_batch = list(random_pool)

    #apply kcenter_greedy over the latent space features
    new_batch = kcenter_greedy(features, curr_batch, args.budget, logger)
    curr_batch = curr_batch+new_batch

    net.train(curr_batch, args.epochs, args.optimizer, args.test_set, learning_rate=args.lr, batch_size=args.bs, momentum=args.momentum, always_test=0)
    logger.info('Training completed')
    net.test(epoch=args.epochs)                

    return 

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--budget', type=int, default=1000, 
        help='Budget for greedy selection or random selection')
    argparser.add_argument('--init-budget', type=int, default=1000,
        help='budget for initial random pool for greedy selection later on') 


    argparser.add_argument('--epochs', type=int, default=50,
        help='Number of training epochs')
    argparser.add_argument('--lr', type=float, default=0.00574,
        help='Learning Rate')
    argparser.add_argument('--bs', type=int, default=32,
        help='Batch Size')
    argparser.add_argument('--momentum', type=float, default=0.9502,
        help='Momentum')
    argparser.add_argument('--optimizer', choices=('SGD', 'ADAM', 'RMS'), default='SGD',
        help='optimizer to use')

    argparser.add_argument('--test-set', choices=('TEST','validate'), default='TEST',
        help='TEST uses standard test-data; validate uses given data as test-data(used in tune.py)')

   
    argparser.add_argument('--verbose', default=5, type=int, choices=(1, 2, 3, 4, 5),
        help='Level of Verbosity')    

    arguments = argparser.parse_args()
    print(arguments)
    main(arguments)
