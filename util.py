import os
import numpy as np
from sklearn.metrics import pairwise_distances


#kcenter_greedy is written based on the code given at "https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py"
def kcenter_greedy(features, subset, budget, logger):

    logger.info('Entered greedy_k_center')
    x = features[subset,:]

    dist = pairwise_distances(features, x, metric='euclidean')
    min_distances = np.min(dist, axis=1).reshape(-1,1)
    new_batch = []
    for _ in range(budget):
        ind = np.argmax(min_distances)
        assert ind not in subset

        x = features[ind,:]
        x = x.reshape(1, -1)
        dist = pairwise_distances(features, x)
        min_distances = np.minimum(min_distances, dist)
        new_batch.append(ind)

    logger.info('Maximum distance from cluster centers is %0.2f', max(min_distances))
    return new_batch

def hinge_loss(features, subset, budget, logger):
    logger.info('Entered hinge-loss function')

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

def max_entropy(features, subset, budget, logger):
    logger.info('Entered max-entropy function')

    # get the features of unlabelled training datapoints
    unlabelled = list(set(np.arange(features.shape[0]))-set(subset))
    features = features[unlabelled]

    # compute the entropy with softmax outputs in each row
    entropy = np.sum(-features*np.log(features), axis=1)

    #sort unlabeled data in decreasing order of entropy and select 'budget' ids
    ids = np.argsort(entropy)[::-1]
    ids = ids[:budget]
    new_batch = np.array(unlabelled)[ids]

    new_batch = list(new_batch)
    return new_batch

def max_variation_ratio(features, subset, budget, logger):
    logger.info('Entered Variation ratio function')

    # get the features of unlabelled training datapoints
    unlabelled = list(set(np.arange(features.shape[0]))-set(subset))
    features = features[unlabelled]

    # sort each row having softmax outputs in increasing order
    features.sort(axis=1)

    # compute 1-max{P(y_hat|x)} (called Variation ratio)
    variation_ratio = 1-features[:,-1]

    # sort unlabeled data in decreasing order of variation-ratio values and select 'budget' ids
    ids = np.argsort(variation_ratio)[::-1]
    ids = ids[:budget]
    new_batch = np.array(unlabelled)[ids]

    new_batch = list(new_batch)
    return new_batch


