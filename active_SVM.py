from sklearn import svm 
import numpy as np 
import torch
import os
import sys
import logging
import argparse
import matplotlib.pyplot as plt 
from math import pi 
import matplotlib as mpl 
from matplotlib.font_manager import FontProperties
font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')

# mpl.rc('font', family='Times New Roman')

def simple_margin(features, subset, budget, logger):

    # get the indices of unlabelled training datapoints
    unlabelled = list(set(np.arange(features.shape[0]))-set(subset))

    # get the features of unlabelled training datapoints
    features = features[unlabelled]

    # pick the points in the increasing order of their distance from estimated boundary
    ids = np.argsort(features)
    ids = ids[:budget]
    new_batch = np.array(unlabelled)[ids]    
    new_batch = list(new_batch)
    return new_batch 

def svm_train_test(train_data, train_labels, test_data, test_labels, kernel, logger):
	# trains an SVM with the chosen kernel on the subset of labeled data
	clf = svm.SVC(kernel=kernel)
	clf.fit(train_data, train_labels)
	outs = clf.predict(test_data)
	test_acc = np.count_nonzero(outs == test_labels)*100/test_data.shape[0]
	return test_acc, clf


def main(args):

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	# load the training data
	root_dir = "./synth-data"
	if (args.dataset == 'poly'):

		data = torch.Tensor(np.load(os.path.join(root_dir,'poly-data-train.npy')))
		labels = torch.Tensor(np.load(os.path.join(root_dir,'poly-label-train.npy')))
		test_data = torch.Tensor(np.load(os.path.join(root_dir,'poly-data-test.npy')))
		test_labels = torch.Tensor(np.load(os.path.join(root_dir,'poly-label-test.npy')))

		# true polynomial decision boundary
		radius = 3*pi
		t = np.linspace(-radius, radius, 500)
		yup = t*(t-pi)*(t+pi)*(t-2*pi)*(t+2*pi)/150

	elif (args.dataset == 'gaussian'):

		data = torch.Tensor(np.load(os.path.join(root_dir,'gaussian-data-train.npy')))
		labels = torch.Tensor(np.load(os.path.join(root_dir,'gaussian-label-train.npy')))
		test_data = torch.Tensor(np.load(os.path.join(root_dir,'gaussian-data-test.npy')))
		test_labels = torch.Tensor(np.load(os.path.join(root_dir,'gaussian-label-test.npy')))

	logger = logging.getLogger("Active-Learning-NN-Train")
	logger.setLevel(10)
	logger.addHandler(logging.StreamHandler(sys.stdout))	

	# get the initial random pool for greedy selection later on
	random_pool = np.arange(labels.shape[0])
	np.random.shuffle(random_pool)
	random_pool = random_pool[:args.init_budget]

	# train the model using the initial random pool of labelled data
	test_acc, clf = svm_train_test(data[random_pool].numpy(), labels[random_pool].numpy(), test_data.numpy(), test_labels.numpy(), args.kernel, logger)
	logger.info('Trained with initial random pool; Test acc:%f', test_acc)

	preds = clf.predict(data.numpy())
	cols = ['r' if(i) else 'g' for i in preds]
	print (labels[random_pool].numpy())
	fig = plt.figure()
	fig.set_size_inches(2.5, 2.1)

	# plots the true polynomial boundary on the figure
	if (args.dataset=='poly'):
		plt.plot(t, yup, color='black')
	
	plt.xlim([-3*pi, 3*pi])
	plt.ylim([-3*pi, 3*pi])

	# plots the training data with colors indicating predicted labels
	plt.scatter(data.numpy()[:,0], data.numpy()[:,1], c=cols, alpha=0.6, s=8)
	# labeled data so far is indicated in blue
	plt.scatter(data[random_pool][:,0], data[random_pool][:,1], color='blue', alpha=0.6, s=10)

	curr_batch = list(random_pool)
	for iter_ in range(args.n_iter):

		# measures the distance from estimated boundary obtained with labeled data so far
		features = clf.decision_function(data.numpy())
		features = np.abs(features)

		# pick the points close to the estimated boundary
		new_batch = simple_margin(features, curr_batch, args.budget//args.n_iter, logger)

		# add this queried dtapoints to the labeled pool and retrain the model
		curr_batch = curr_batch+new_batch
		test_acc, clf = svm_train_test(data[curr_batch].numpy(), labels[curr_batch].numpy(), test_data.numpy(), test_labels.numpy(), args.kernel, logger)
		logger.info('Collected %d/%d points so far; Test acc:%f',len(curr_batch), args.budget+args.init_budget, test_acc)

		preds = clf.predict(data.numpy())
		cols = ['r' if(i) else 'g' for i in preds]

		# Queried points are indicated by 'black' squares on figure
		plt.scatter(data[new_batch][:,0], data[new_batch][:,1], color='black', alpha=0.8, marker='s', s=8)
		plt.title('Labelled:%s, Queried:%s' % (str(len(curr_batch)-len(new_batch)), str(len(new_batch))), fontproperties=font)
		# fig.savefig('figures/iter-'+str(iter_)+'-'+args.dataset+'-'+args.figstring+'.eps',format='eps',bbox_inches='tight')

		fig = plt.figure()
		fig.set_size_inches(2.5, 2.1)
		if (args.dataset=='poly'):
			plt.plot(t, yup, color='black')
		plt.xlim([-3*pi, 3*pi])
		plt.ylim([-3*pi, 3*pi])
		plt.scatter(data.numpy()[:,0], data.numpy()[:,1], c=cols, alpha=0.6, s=8)
		plt.scatter(data[curr_batch][:,0], data[curr_batch][:,1], color='blue', alpha=0.6, s=10)
			
		
	plt.title('Labelled:%s' %(str(len(curr_batch))), fontproperties=font)
	# fig.savefig('figures/iter-'+str(args.n_iter)+'-'+args.dataset+'-'+args.figstring+'.eps',format='eps',bbox_inches='tight')
	logger.info('Completed %d iterations', args.n_iter)
	plt.show()
	return

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', choices=('gaussian','poly'), default='poly',
    	help='Dataset to use')
    argparser.add_argument('--budget', type=int, default=10, 
        help='Budget for greedy selection or random selection')
    argparser.add_argument('--init-budget', type=int, default=10,
        help='budget for initial random pool for greedy selection later on')  
    argparser.add_argument('--n-iter', type=int, default=1,
        help='Number of iterations of greedy selection')
    argparser.add_argument('--seed', type=int, default=1,
    	help='seed for randomness')  
    argparser.add_argument('--kernel', choices=('rbf', 'linear'), default='rbf',
    	help='kernel for SVM')
    argparser.add_argument('--figstring', type=str, default=None)

    arguments = argparser.parse_args()
    print(arguments)
    main(arguments)



