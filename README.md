
## Overview

This repository contains an op-for-op PyTorch reimplementation of [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).


## DataSet

This implemenation uses the [ILSVRC 2012 dataset](http://www.image-net.org/challenges/LSVRC/2012/), also known as the 'ImageNet 2012 dataset'.
The data size is dreadfully large (138G!), but this amount of large-sized dataset is required for successful training of AlexNet.
Testing with [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/) or [MNIST](http://yann.lecun.com/exdb/mnist/) could not be done due to their smaller feature sizes (images do not fit the input size 227 x 227).

After downloading the dataset file (i.e., `ILSVRC2012_img_train.tar`), use `extract_imagenet.sh` to extract the entire dataset.
