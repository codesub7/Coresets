import numpy as np 
import matplotlib.pyplot as plt 
import numpy.random as random
from math import pi as pi 
import sys 
import os 

train_labels = []
s1 = np.random.multivariate_normal(np.array([1.2*pi,0]), np.array([[1*pi,0],[0,1*pi]]), 500)
train_labels.extend([1]*500)
s2 = np.random.multivariate_normal(np.array([-1.2*pi,0]), np.array([[1*pi,0],[0,1*pi]]), 500)
train_labels.extend([0]*500)
train_data = np.concatenate((s1, s2), axis=0)


fig1 = plt.figure()

viz_color = ['r' if i else 'g' for i in train_labels]
plt.scatter(train_data[:,0], train_data[:,1], c=viz_color, alpha=0.8)
plt.title('gaussian-data')
plt.xlim([-3*pi, 3*pi])
plt.ylim([-3*pi, 3*pi])
# fig1.savefig('plots_dir/gaussian-data-train.pdf')
# fig1.savefig('plots_dir/gaussian-data-train.png')


# Test data
test_labels = []
s1 = np.random.multivariate_normal(np.array([1.2*pi,0]), np.array([[1*pi,0],[0,1*pi]]), 500)
test_labels.extend([1]*500)
s2 = np.random.multivariate_normal(np.array([-1.2*pi,0]), np.array([[1*pi,0],[0,1*pi]]), 500)
test_labels.extend([0]*500)
test_data = np.concatenate((s1, s2), axis=0)

viz_color = ['r' if i else 'g' for i in test_labels]
plt.figure()
plt.scatter(test_data[:,0], test_data[:,1], c=viz_color, alpha=0.8)
plt.xlim([-3*pi, 3*pi])
plt.ylim([-3*pi, 3*pi])
# # Saving data
# np.save('synth-data/gaussian-data-train.npy', train_data)
# np.save('synth-data/gaussian-label-train.npy', train_labels)
# np.save('synth-data/gaussian-data-test.npy', test_data)
# np.save('synth-data/gaussian-label-test.npy', test_labels)
plt.show()
