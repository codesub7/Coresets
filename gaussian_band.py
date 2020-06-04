import numpy as np 
import matplotlib.pyplot as plt 
import numpy.random as random
from math import pi as pi 
import sys 
import os 
import logging
from sklearn.linear_model import LogisticRegression
import argparse
from sklearn.metrics import pairwise_distances
import matplotlib as mpl 

mpl.rc('font', family='Times New Roman')

def logistic_train_test(train_data, train_labels, test_data, test_labels, logger):
	clf = LogisticRegression(random_state=0).fit(train_data, train_labels)
	outs = clf.predict(test_data)
	test_acc = np.count_nonzero(outs == test_labels)*100/test_data.shape[0]
	return test_acc, clf

def hinge_loss(features, subset, budget, logger):

    # get the indices of unlabelled training datapoints
    unlabelled = list(set(np.arange(features.shape[0]))-set(subset))

    # get the features of unlabelled training datapoints
    features = features[unlabelled]

    # sort each row having softmax outputs in increasing order
    features.sort(axis=1)

    # difference between max of softmax and second highest of softmax
    hinge_loss = features[:,-1]-features[:,-2]

    # sort the hingeloss in increasing order and select top 'budget' ids
    ids = np.argsort(hinge_loss)
    ids = ids[:budget]
    new_batch = np.array(unlabelled)[ids]
    
    new_batch = list(new_batch)
    return new_batch 

#kcenter_greedy is written based on the code given at "https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py"
def kcenter_greedy(features, subset, budget, logger):

    x = features[subset,:]

    dist = pairwise_distances(features, x, metric='euclidean')
    min_distances = np.min(dist, axis=1).reshape(-1,1)
    new_batch = []
    for _ in range(budget):
        ind = np.argmax(min_distances)
        x = features[ind,:]
        x = x.reshape(1, -1)
        dist = pairwise_distances(features, x)
        min_distances = np.minimum(min_distances, dist)
        new_batch.append(ind)

    return new_batch, max(min_distances)

def points_near_boundary(subset, train_data, eps):
	queried = train_data[subset]
	t = (queried[:,0] > -eps) & (queried[:,0] < eps)
	return np.count_nonzero(t)


argparser = argparse.ArgumentParser()
argparser.add_argument('--init-budget', type=int, default=10,
	help='size of initial random pool')
argparser.add_argument('--eps', type=float, default=0.2*pi)
argparser.add_argument('--seed', type=int, default=1)
args = argparser.parse_args()


np.random.seed(args.seed)

root_dir = "./synth-data"
train_data = np.load(os.path.join(root_dir,'gaussian-data-train.npy'))
train_labels = np.load(os.path.join(root_dir,'gaussian-label-train.npy'))
test_data = np.load(os.path.join(root_dir,'gaussian-data-test.npy'))
test_labels = np.load(os.path.join(root_dir,'gaussian-label-test.npy'))

logger = logging.getLogger("Active-Learning-NN-Train")
logger.setLevel(10)
logger.addHandler(logging.StreamHandler(sys.stdout))

budgets = np.linspace(10,train_data.shape[0]-args.init_budget-1,30)
rand = []
kcenter = []

repeat = 5
for r in np.arange(repeat):

    rand_Nb = []
    kcenter_Nb = [] 

    # for each budget, we compute the number of points falling in narrow band around true decision boundary (when random selection/kcenter is used)
    for budget in budgets:

        budget = int(budget)
        #Random selection
        subset = np.arange(len(train_labels))
        np.random.shuffle(subset)
        subset = subset[:(budget+args.init_budget)]
        rand_Nb.append(points_near_boundary(subset, train_data, args.eps))

        #Initial random pool for greedy approaches
        random_pool = np.arange(len(train_labels))
        np.random.shuffle(random_pool)
        random_pool = random_pool[:args.init_budget]

        test_acc, clf = logistic_train_test(train_data[random_pool], np.array(train_labels)[random_pool], test_data, np.array(test_labels), logger)
        # 'features' represent softmax outputs corresponding to two classes
        features = clf.predict_proba(train_data)

        ##Kcenter
        new_batch,_  = kcenter_greedy(features, list(random_pool), budget, logger)
        kcenter_Nb.append(points_near_boundary(new_batch+list(random_pool), train_data, args.eps))

    rand.append(rand_Nb)
    kcenter.append(kcenter_Nb)

m_rand = np.array(rand).mean(axis=0)
m_kcenter = np.array(kcenter).mean(axis=0)

s_rand = np.array(rand).std(axis=0)
s_kcenter = np.array(kcenter).std(axis=0)

budgets = np.array(budgets)+args.init_budget

fig1 = plt.figure()
fig1.set_size_inches(2.5, 2.1)
plt.errorbar(budgets, m_rand, s_rand, fmt='-^', label='Random',markersize='4')
plt.errorbar(budgets, m_kcenter, s_kcenter, fmt='-^', label='Kcenter',markersize='4')
plt.legend(loc='lower right',prop={'size':6})
plt.xlim([0,train_data.shape[0]])
plt.ylim([0,50])
plt.xlabel('Budget')
plt.ylabel('Number of points in $N_b$')
plt.grid()
fig1.savefig('gaussian-band.eps', format='eps', bbox_inches = 'tight')
plt.show()
