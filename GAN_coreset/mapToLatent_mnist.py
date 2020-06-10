import tensorflow as tf 
import numpy as np 
import pickle

#Import dataset
mnist = tf.keras.datasets.mnist

tfd = tf.contrib.distributions

batch_size = 32
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print ('shape:',x_train.shape)
n_data = x_train.shape[0]

# Convert dataset to the tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices(x_train.astype(np.float32))
train_dataset = train_dataset.repeat(2)
train_dataset = train_dataset.batch(batch_size=batch_size)

x_train_samples = train_dataset.make_one_shot_iterator().get_next()
x_train_samples = tf.reshape(x_train_samples, [batch_size, 28, 28, 1])
x_train_samples = tf.image.resize_bilinear(x_train_samples, size=(24, 24))

count = 0
beta_ph = 1.0
sess = tf.Session()

#Import trained GAN
new_saver = tf.train.import_meta_graph('checkpoints_lr0.001/mnist-glow.ckpt.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./checkpoints_lr0.001/'))

#get graph tensors to input images and extract latent-space features
graph = tf.get_default_graph()
images = graph.get_tensor_by_name('features_batch:0')
beta = graph.get_tensor_by_name('Placeholder:0')
log_prob_X = graph.get_tensor_by_name('logPx/add_1:0')
latent_space = graph.get_tensor_by_name('latent_space/concat:0')
log_prob_z = graph.get_tensor_by_name('logPx/add:0')

# for op in graph.get_operations():
# 	if ('logPx' in op.name) :
# 		print (op.name, op.type)

z_array = []
logPx_array = []
logPz_array = []

for i in range(n_data//batch_size + 1):
	imbatch = sess.run(x_train_samples)	
	l_space, logPx, logPz = sess.run([latent_space, log_prob_X, log_prob_z], {images: imbatch, beta: beta_ph})
	z_array.append(l_space)	
	logPx_array.append(logPx)
	logPz_array.append(logPz)

	count += batch_size
	if (count >= n_data):
		break

z_array = np.array(z_array).reshape((-1,l_space.shape[1]))
logPx_array = np.array(logPx_array).flatten()
logPz_array = np.array(logPz_array).flatten()

z_array = z_array[:n_data]
logPx_array = logPx_array[:n_data]
logPz_array = logPz_array[:n_data]

#saving latent-space features and input distribution in the original space and latent space
data_dict = {}
data_dict['z'] = z_array
data_dict['logPx'] = logPx_array
data_dict['logPz'] = logPz_array

with open('mnist-latentspace.pickle', 'wb') as f:
	pickle.dump(data_dict, f)

print('File saved!')
