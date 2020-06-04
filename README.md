## Systematic Exploration of Algorithms for Selecting Coresets
* [Nets](https://github.com/codesub7/Coresets/tree/master/Nets): It has the code for VGG16 models for CIFAR10, SVHN datasets along with a base class `NeuralNetwork` implementation.
 
* [util.py](https://github.com/codesub7/Coresets/blob/master/util.py) has the code for k-center and various uncertainty based approaches.

* [active_svm.py](https://github.com/codesub7/Coresets/blob/master/active_SVM.py) has the code to perform experiments on synthetic data mentioned in our paper. 

* [synth-data](https://github.com/codesub7/Coresets/tree/master/synth-data) has two different synthetic datasets generated using [generate_gaussian_data.py](https://github.com/codesub7/Coresets/blob/master/generate_gaussian_data.py) and [generate_poly_data.py](https://github.com/codesub7/Coresets/blob/master/generate_poly_data.py) scripts.

* To create a conda environment for our project `conda env create -f environment.yaml`

* To run training on SVHN with random subset

     `python train.py --dataset svhn --method random --budget 5000 --epochs 50 --lr 0.00247 --bs 32 --momentum 0.929 --optimizer SGD`
* To run training on SVHN with a iterative subset selection algorithm, there are these greedy approaches: 1)Kcenter 2)Hingeloss 3)Entropy 4)Variation ratio. Each of these approaches spends the given budget over multiple iterations.

     `python train.py --dataset svhn --method greedy --greedy-approach kcenter --init-budget 5000 --budget 2500 --n-iter 1  --epochs 50 --lr 0.00247 --bs 32 --momentum 0.929 --optimizer SGD`
* To get the results on Gaussian synthetic data-setting given in the paper

    `python active_svm.py --dataset gaussian --init-budget 5 --budget 20 --n-iter 1 --seed 1 --kernel linear`
* To reproduce the results on data setting with non-linear decision boundary

   `python active_svm.py --dataset poly --init-budget 10 --budget 20 --n-iter 1 --seed 4 --kernel rbf`
   
   `python active_svm.py --dataset poly --init-budget 10 --budget 20 --n-iter 5 --seed 4 --kernel rbf`

* For help

     `python train.py  --help`

* We perform hyperparameter tuning with `tune.py` using [hyperopt.](https://github.com/hyperopt/hyperopt/wiki)(*modify line 14 in tune.py to point to the root directory of clone of this repository*)

### How to run distributed hyperopt?
* [Install MongoDB](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/) on all the nodes
* Start a mongod process in root node as mentioned below

     `mongod --dbpath . --port 1234 --directoryperdb --journal --bind_ip_all`
* In another terminal on the root node, cd to the git repository's root directory and run tune.py as:

     `python tune.py --dataset cifar10 --epochs 25 --optimizer SGD --test-set validate --train-file <train_file> --validate-file <validate_file> --trials 20 --parallel`
where test-set='validate' indicates that we use a held-out set for computing the loss corresponding to each hyperparam config in hyperopt jobs.
* On slave nodes, again cd to git repository's root directory, then run hyperopt-mongo-worker as follows (*using appropriate db name and location*):

     `hyperopt-mongo-worker --mongo=10.10.1.4:1234/cifar10_random_0_none_5000_30_5`
     
**Note:** install `dill` module for the slave nodes to be able to download and unpickle the serialized objects from master node.
