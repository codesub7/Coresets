[trainGAN_mnist.py](https://github.com/codesub7/Coresets/blob/master/GAN_coreset/trainGAN_mnist.py) can be used to train the flow-based generative model.

[mapToLatent_mnist.py](https://github.com/codesub7/Coresets/blob/master/GAN_coreset/mapToLatent_mnist.py) uses the trained generative model to map the mnist image data to its latent space.

[GAN_coreset.py](https://github.com/codesub7/Coresets/blob/master/GAN_coreset/GAN_coreset.py) implements the coreset selection using the latent space mappings of the mnist training data.
