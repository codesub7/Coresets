'''This code to train GAN for MNIST is taken from:
https://github.com/kmkolasinski/deep-learning-notes/tree/master/seminars/2018-10-Normalizing-Flows-NICE-RealNVP-GLOW
'''
import numpy as np
import tensorflow as tf
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#Import dataset
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train.shape, x_test.shape, y_train.shape, y_test.shape

# Convert dataset to the tf.data.Dataset
import utils
import nets
import flow_layers as fl

batch_size = 32
train_dataset = utils.numpy_array_to_dataset(array=x_train, batch_size=batch_size)

x_train_samples = train_dataset.make_one_shot_iterator().get_next()

x_train_samples = tf.reshape(x_train_samples, [batch_size, 28, 28, 1])
x_train_samples = tf.image.resize_bilinear(x_train_samples, size=(24, 24))
x_train_samples

tf.set_random_seed(0)
sess = tf.InteractiveSession()
# Check shapes
x_train_samples.eval().shape, x_train_samples.eval().max()
# Build Flow with Resnet blocks
nn_template_fn = nets.ResentTemplate(
    units_factor=6, num_blocks=3
)

layers, actnorm_layers = nets.create_simple_flow(
    num_steps=5, 
    num_scales=3, 
    template_fn=nn_template_fn
)
# create model
images = x_train_samples
images = tf.placeholder(tf.float32, shape= (32, 24, 24, 1), name= 'features_batch')
flow = fl.InputLayer(images)
model_flow = fl.ChainLayer(layers)
output_flow = model_flow(flow, forward=True)
# Prepare output tensors
y, logdet, z = output_flow
output_flow
# Build loss function and prior distributions for p(y) and p(z)
tfd = tf.contrib.distributions

beta_ph = tf.placeholder(tf.float32, [])

y_flatten = tf.reshape(y, [batch_size, -1])
z_flatten = tf.reshape(z, [batch_size, -1])

with tf.variable_scope("latent_space"):
    latent_space = tf.concat([y_flatten, z_flatten], 1)    

prior_y = tfd.MultivariateNormalDiag(loc=tf.zeros_like(y_flatten), scale_diag=beta_ph * tf.ones_like(y_flatten))
prior_z = tfd.MultivariateNormalDiag(loc=tf.zeros_like(z_flatten), scale_diag=beta_ph * tf.ones_like(z_flatten))

log_prob_y =  prior_y.log_prob(y_flatten)
log_prob_z =  prior_z.log_prob(z_flatten)

prior_z
# The MLE loss
with tf.variable_scope("logPx"):
    log_prob_X = log_prob_y + log_prob_z + logdet

loss = - tf.reduce_mean(log_prob_X)
# The L2 regularization loss
trainable_variables = tf.trainable_variables() 
l2_reg = 0.001 
l2_loss = l2_reg * tf.add_n([ tf.nn.l2_loss(v) for v in trainable_variables])
# Debug model, print variables
total_params = 0
for k, v in enumerate(trainable_variables):
    num_params = np.prod(v.shape.as_list())
    total_params += num_params
    print(f"[{k:4}][{num_params:6}] {v.op.name[:96]}")

print(f"total_params: {total_params}")
# Total loss -logp(x) + l2_loss
sess.run(tf.global_variables_initializer())

total_loss = l2_loss + loss

imbatch = sess.run(x_train_samples)
l2_loss.eval(feed_dict={beta_ph: 1.0, images: imbatch}), loss.eval(feed_dict={beta_ph: 1.0, images: imbatch})

# Create backward flow to generate samples
sample_y_flatten = prior_y.sample()
sample_y = tf.reshape(sample_y_flatten, y.shape.as_list())
sample_z = tf.reshape(prior_z.sample(), z.shape.as_list())
sampled_logdet = prior_y.log_prob(sample_y_flatten)

inverse_flow = sample_y, sampled_logdet, sample_z
sampled_flow = model_flow(inverse_flow, forward=False)

x_flow_sampled, _, _ = sampled_flow

x_flow_sampled.eval({beta_ph: 1.0}).shape
# Define optimizer and learning rate
lr_ph = tf.placeholder(tf.float32)

optimizer = tf.train.AdamOptimizer(lr_ph)
train_op = optimizer.minimize(total_loss)
# Initialize Actnorms using DDI
sess.run(tf.global_variables_initializer())
nets.initialize_actnorms(
    sess, x_train_samples, images,
    feed_dict_fn=lambda: {beta_ph: 1.0},
    actnorm_layers=actnorm_layers,
    num_steps=50,
)
# Train model, define metrics and trainer
metrics = utils.Metrics(100, metrics_tensors={"total_loss": total_loss, "loss": loss, "l2_loss": l2_loss})
plot_metrics_hook = utils.PlotMetricsHook(metrics, step=1000)

imbatch = sess.run(x_train_samples)
sess.run(train_op, feed_dict={lr_ph: 0.0, beta_ph: 1.0, images: imbatch})
total_loss.eval(feed_dict={lr_ph: 0.0, beta_ph: 1.0, images: imbatch})
# Initial samples from model
# x_samples_np = x_flow_sampled.eval(feed_dict={lr_ph: 0.0, beta_ph: 1.0})
# utils.plot_4x4_grid(x_samples_np, shape=x_samples_np.shape[1:3])
# Train model with lr=0.005
utils.trainer(
    sess, x_train_samples, images,
    num_steps=100000, 
    train_op=train_op, 
    feed_dict_fn=lambda: {lr_ph: 0.0001, beta_ph: 1.0}, 
    metrics=[metrics], 
    hooks=[plot_metrics_hook]
)
saver = tf.compat.v1.train.Saver()
saver.save(sess, "./checkpoints_lr"+str(0.0001)+"/" + 'mnist-glow' + ".ckpt")

