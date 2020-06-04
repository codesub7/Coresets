import sys
import logging
import torch
import numpy as np
import argparse

from Nets.CIFAR10 import CIFAR10VGG16
from Nets.SVHN import SVHNVGG16

from util import *
import torch.nn.functional as F
import time
import pickle

def main(args):

    logger = logging.getLogger("Active-Learning-NN-Train")
    logger.setLevel((6 - args.verbose) * 10)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    if args.dataset == 'cifar10':
        net = CIFAR10VGG16(logger, './data')
    elif args.dataset == 'svhn':
        net = SVHNVGG16(logger, './data')

    # it creates batches of training data with 1000 points in each batch 
    net.read_dataset(1000)
    net.build_model()
    logger.info('Completed reading dataset; built NN model')

    # select the random subset for 'Random selection' approach
    if(args.method == 'random'):
        subset = np.arange(len(net.train_data))
        np.random.shuffle(subset)
        subset = subset[:args.budget]
        logger.info('Random subset selected')

        # train the model with above selected subset; 
        # When always_test=0, model will not be evaluated at the end of every training epoch to save time 
        # set always_test=1 to evaluate the model with test set at the end of every epoch
        net.train(subset, args.epochs, args.optimizer, args.test_set, learning_rate=args.lr, batch_size=args.bs, momentum=args.momentum, always_test=0)
        logger.info('Finished training with random selection')

        # evaluate the trained model using test data 
        net.test(epoch=args.epochs)
        curr_batch = list(subset)

    if (args.method == 'greedy'):

        # get the initial random pool for greedy selection later on
        if (args.random_pool is not None):
            random_pool = np.load(args.random_pool)
            logger.info('Loaded initial random pool for greedy selection, size:%d', len(list(random_pool)))
        else:
            random_pool = np.arange(len(net.train_data))
            np.random.shuffle(random_pool)
            random_pool = random_pool[:args.init_budget]
            logger.info('Random subset selected')

        # train the model using the initial random pool of labelled data
        if (args.saved_model is not None):
            net.model = torch.load(args.saved_model)
        else:
            net.train(random_pool, args.epochs, args.optimizer, args.test_set, learning_rate=args.lr, batch_size=args.bs, momentum=args.momentum, always_test=0)
            logger.info('Trained with initial random pool')
        net.test(epoch=args.epochs)
        
        # spend the budget for greedy selection over multiple iterations
        curr_batch = list(random_pool)
        for iter_ in range(args.n_iter):
            logger.info('%d iteration: Computing activation outputs...', iter_)
            net.model.eval()
            with torch.no_grad():
                features = []
                targets = []
                for ipt, tgt in net.train_loader:
                    ipt, tgt = ipt.to(net.device), tgt.to(net.device)

                    # For kcenter, pre-softmax outputs are used for greedy selection
                    # For all remaining uncertainty approaches, softmax outputs are used for greedy selection
                    if (args.greedy_approach == 'kcenter'):
                        features.append(net.model(ipt).numpy())
                    else:
                        features.append(F.softmax(net.model(ipt), dim=1).numpy())
                    targets.append(tgt.numpy())
                features = np.vstack(features)
                targets = np.hstack(targets)

            # storing logits and true labels to visualize tsne embeddings for later use
            if (args.store_logits):
                data_dict = {}
                data_dict['features'] = features
                data_dict['targets'] = targets

                with open(args.logit_file, 'wb') as f:
                    pickle.dump(data_dict, f)

            # choose the points to be queried in this iteration based on 'features' obtained from partially trained model 
            start_time = time.time()
            if (args.greedy_approach == 'kcenter'):
                logger.info('%d iteration: Entering kcenter greedy', iter_)
                new_batch = kcenter_greedy(features, curr_batch, args.budget//args.n_iter, logger)
                logger.info('%d iteration: completed kcenter greedy', iter_)

            elif (args.greedy_approach == 'hingeloss'):
                logger.info('%d iteration: hingeloss based greedy selection', iter_)
                new_batch = hinge_loss(features, curr_batch, args.budget//args.n_iter, logger)
                logger.info('%d iteration: completed hinge-loss based selection', iter_)

            elif (args.greedy_approach == 'maxentropy'):
                logger.info('%d iteration: entropy maximizing selection', iter_)
                new_batch = max_entropy(features, curr_batch, args.budget//args.n_iter, logger)
                logger.info('%d iteration: completed entropy based selection', iter_)

            elif (args.greedy_approach == 'variationratio'):
                logger.info('%d iteration: variation ratio maximizing selection', iter_)
                new_batch = max_variation_ratio(features, curr_batch, args.budget//args.n_iter, logger)
                logger.info('%d iteration: completed variation ratio based selection', iter_)

            logger.info('TIME TAKEN: %f', time.time()-start_time)

            # re-train the model with the current batch and next batch of labelled data
            curr_batch = curr_batch+new_batch
            net.train(curr_batch, args.epochs, args.optimizer, args.test_set, learning_rate=args.lr, batch_size=args.bs, momentum=args.momentum, always_test=0)
            logger.info('%d iteration: Training completed', iter_)
            net.test(epoch=args.epochs)

        logger.info('Completed %d iterations', args.n_iter)

    return 

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', choices=('cifar10', 'svhn'), default='cifar10',
        help='Dataset to use')
    argparser.add_argument('--method', choices=('greedy', 'random'), default='random',
        help='Method to use')
    argparser.add_argument('--greedy-approach', choices=('kcenter', 'hingeloss', 'maxentropy', 'variationratio'), default='kcenter',
        help='Approach to use for subset selection in every iteration')
    argparser.add_argument('--budget', type=int, default=10, 
        help='Budget for greedy selection or random selection')
    argparser.add_argument('--init-budget', type=int, default=10,
        help='budget for initial random pool for greedy selection later on')  
    argparser.add_argument('--n-iter', type=int, default=1,
        help='Number of iterations of greedy selection')


    argparser.add_argument('--saved-model', type=str, default=None,
        help='model trained over initial random pool for greedy selection later on (Optional)')
    argparser.add_argument('--random-pool', type=str, default=None,
        help='file name of initial random pool for greedy selection later on (Optional)')


    argparser.add_argument('--epochs', type=int, default=50,
        help='Number of training epochs')
    argparser.add_argument('--lr', type=float, default=1e-2,
        help='Learning Rate')
    argparser.add_argument('--bs', type=int, default=32,
        help='Batch Size')
    argparser.add_argument('--momentum', type=float, default=0.9,
        help='Momentum')
    argparser.add_argument('--optimizer', choices=('SGD', 'ADAM', 'RMS'), default='SGD',
        help='optimizer to use')

    argparser.add_argument('--test-set', choices=('TEST','validate'), default='TEST',
        help='TEST uses standard test-data; validate uses given data as test-data(used in tune.py)')

    
    argparser.add_argument('--store-logits', action='store_true',
        help='Store the logits resulting from the model trained on initial random pool')
    argparser.add_argument('--logit-file', type=str, default=None,
        help='file name to store the logits')
    argparser.add_argument('--verbose', default=5, type=int, choices=(1, 2, 3, 4, 5),
        help='Level of Verbosity')    

    arguments = argparser.parse_args()
    print(arguments)
    main(arguments)
