import numpy as np 
from sklearn.manifold import TSNE
import pickle
import matplotlib.pyplot as plt 

#file with logits and true labels (can obtained by running train.py)
with open('logit_data_5k.pkl','rb') as f:
	data_dict = pickle.load(f) 

features = data_dict['features']
targets = data_dict['targets']

lowd_x = TSNE(n_components=2, verbose=10).fit_transform(features)
# np.save('lowd_features.npy', lowd_x)

# colors for each of the 10 labels
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'brown', 'orange', 'purple']
cols = [colors[t] for t in targets]
# lowd_x = np.load('lowd_features.npy')


# choosing 5000 random points for non-congested visualization of tsne embeddings
random_pool = np.arange(len(cols))
np.random.shuffle(random_pool)
random_pool = random_pool[:5000]

cols = np.array(cols)[random_pool]
lowd_x = lowd_x[random_pool, :]

# plotting low dimensional data obtained from tsne approach
plt.scatter(lowd_x[:,0], lowd_x[:,1], c=cols)
# plt.savefig('svhn_tsne.png')
# plt.savefig('svhn_tsne.pdf')

plt.show()