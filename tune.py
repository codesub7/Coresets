import os
import sys
import logging
from functools import partial
import numpy as np
import torch
from hyperopt import fmin, tpe, hp, STATUS_OK
from hyperopt.mongoexp import MongoTrials
import argparse


def run(args, space):
	import sys
	sys.path.append("/workspace/Coresets-for-activelearning")
	import logging
	from functools import partial

	import numpy as np
	import torch

	from hyperopt import fmin, tpe, hp, STATUS_OK
	from hyperopt.mongoexp import MongoTrials

	from Nets.CIFAR10 import CIFAR10VGG16
	from Nets.SVHN import SVHNVGG16

	logger = logging.getLogger("Active-Learning-NN-Train")
	logger.setLevel((6 - args.verbose) * 10)
	logger.addHandler(logging.StreamHandler(sys.stdout))

	if args.dataset == 'cifar10':
		net = CIFAR10VGG16(logger, './data')
	elif args.dataset == 'svhn':
		net = SVHNVGG16(logger, './data')

	net.read_dataset()
	logger.info('Completed reading dataset; built NN model')

	# get training and validation data for hyperparam tuning
	if (args.train_file and args.validate_file):
		train_subset = np.load(args.train_file)
		validate_subset = np.load(args.validate_file)
	else:
		subset = np.arange(len(net.train_data))
		np.random.shuffle(subset)
		train_subset = subset[:args.train_size]
		validate_subset = subset[args.train_size:]
		validate_subset = validate_subset[:args.validation_size] 

	name = args.dataset
	logger.info('Setup done!')
	logger.info('Params', extra={'props': space})

	print(space)

	# create dataloader using validation subset. 
	# This will be used to evaluate the model corresponding to each hyperparam config
	net.validate_loader = net.create_dataloader(validate_subset)

	# train the model with above selected train_subset; 
    # When always_test=0, model will not be evaluated at the end of every training epoch to save time 
    # set always_test=1 to evaluate the model with test set at the end of every epoch
	net.train(train_subset, args.epochs, args.optimizer, args.test_set, always_test=0, **space)
	test_acc,_ = net.test(epoch=args.epochs)
	return {'loss': 100.0 - test_acc,'status': STATUS_OK}

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', choices=('cifar10', 'svhn'),
        help='Dataset to use', default='cifar10')
	parser.add_argument('--epochs', type=int,
		help='Number of epochs to run for', default=25)
	parser.add_argument('--optimizer', choices=('SGD', 'ADAM', 'RMS'), default='SGD',
        help='optimizer to use')
	parser.add_argument('--test-set', choices=('TEST','validate'), default='validate',
        help='TEST uses standard test-data; validate uses given data as test-data(used in tune.py)')

	parser.add_argument('--train-file', type=str, default=None,
		help='file from which we load train data for hyperparam tuning')
	parser.add_argument('--validate-file', type=str, default=None,
		help='file from which we load validation data to compute loss w.r.t each chosen hyperparam config')
	parser.add_argument('--train-size', type=int, default=2500,
		help='size of the train data for hyperparam tuning')
	parser.add_argument('--validation-size', type=int, default=2500,
		help='size of the validation data for hyperparam tuning')

	parser.add_argument('--trials', type=int,
		help='Number of hyperopt trials')
	parser.add_argument('--parallel', action='store_true',
		help='Use this option to perform parallel search')

	parser.add_argument('--verbose', default=5, type=int, choices=(1, 2, 3, 4, 5),
        help='Level of Verbosity')
	args = parser.parse_args()

	# represent search space with using hyperopt
	SPACE = {
		'learning_rate': hp.loguniform('learning_rate', np.log(1e-3), np.log(0.1)),
		'batch_size': hp.choice('batch_size', [32, 64, 128, 256, 1024]),
		'momentum': hp.uniform('momentum', 0.9, 0.999)
	}

	# details about parallel search with hyperopt and mongod can be found here: 
	# https://github.com/hyperopt/hyperopt/wiki/Parallelizing-Evaluations-During-Search-via-MongoDB
	# fmin usage can be found here: https://github.com/hyperopt/hyperopt/wiki/FMin
	if args.parallel:
		trials = MongoTrials('mongo://128.105.144.224:1234/mongodb_%s/jobs' % (args.dataset))
		print(trials)

		best = fmin(fn=partial(run, args),
					space=SPACE,
					algo=tpe.suggest,
					trials=trials,
					max_evals=args.trials)
	else:
		best = fmin(fn=partial(run, args),
					space=SPACE,
					algo=tpe.suggest,
					max_evals=args.trials)
	print('Best hyperparams:', best)
	print(trials.losses())
