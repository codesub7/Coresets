import numpy as np 
import matplotlib.pyplot as plt 
import numpy.random as random
from math import pi as pi 
import sys 
import os 

# Using the sin data provided in "https://github.com/Stanford-ILIAD/DPP-Batch-Active-Learning" to generate poly-data:
root_dir = './synth-data'
sin_data_train = np.load(os.path.join(root_dir,'sin-data-train.npy'))
sin_labels_train = np.load(os.path.join(root_dir,'sin-label-train.npy'))
sin_data_test = np.load(os.path.join(root_dir,'sin-data-test.npy'))
sin_labels_test = np.load(os.path.join(root_dir,'sin-label-test.npy'))

radius = 3*pi

#creating boundary using a 5th order polynomial
t = np.linspace(-radius, radius, 500)
yup = t*(t-pi)*(t+pi)*(t-2*pi)*(t+2*pi)/150

fig1 = plt.figure()
plt.plot(t, yup, color='black')

# Train data
train_labels = []
for (x1,x2) in sin_data_train:
	if (x2 > x1*(x1-pi)*(x1+pi)*(x1-2*pi)*(x1+2*pi)/150):
		train_labels.append(1)
	else:
		train_labels.append(0)

viz_color = ['r' if i else 'g' for i in train_labels]

plt.scatter(sin_data_train[:,0], sin_data_train[:,1], c=viz_color, alpha=0.8)
plt.xlim([-3*pi, 3*pi])
plt.ylim([-3*pi, 3*pi])
plt.title('poly-data')
# fig1.savefig('plots_dir/poly-data-train.pdf')
# fig1.savefig('plots_dir/poly-data-train.png')


# Test data
test_labels = []
for (x1,x2) in sin_data_test:
	if (x2 > x1*(x1-pi)*(x1+pi)*(x1-2*pi)*(x1+2*pi)/150):
		test_labels.append(1)
	else:
		test_labels.append(0)

viz_color = ['r' if i else 'g' for i in test_labels]
plt.figure()
plt.scatter(sin_data_test[:,0], sin_data_test[:,1], c=viz_color, alpha=0.8)

# Saving data
# np.save('synth-data/poly-data-train.npy', sin_data_train)
# np.save('synth-data/poly-label-train.npy', train_labels)
# np.save('synth-data/poly-data-test.npy', sin_data_test)
# np.save('synthdata/poly-label-test.npy', test_labels)
plt.show()
