# CUDA-Vision-PyTorch
Contains pytorch Assignments on the following topics:
1. Logistic Regression Classifier.
2. Regularization(L1, L2 and Dropout) with different Optimizers and activation functions.
3. CNN with visualization of activation maps and kernels.
4. DenseNet finetuning for humanoid robot classification.
5. Convolutional Autoencoder and CAE latent space for Classification.

# Vision Systems Lab: Learning Computer Vision on GPUs using Deep Learning Models
Implemented as a part of the CudaVision Lab at University of Bonn (Winter Semester 2019)

Contributors: Saba Khan, [Tejas Morbagal Harish](https://github.com/TejasMorbagal/)

Code repository contains various deep learning models implemented using PyTorch, and Python as Jupyter notebooks. 
Assignment folders contain the respective jupyter notebooks along with output images. 

### Assignment 1: Logistic Regression Classifier using pure python/numpy
Contains a 2 layer network implementation for Logistic Regression Classifier using pure python/numpy, using softmax activation function and mean squared error. The network is trained on MNIST data.

### Assignment 2: Logistic Regression Classifier dataset using PyTorch
Contains the implementation of a Logistic Regression Classifier with CIFAR-10 dataset using PyTorch, along with the confusion matrix and the learning curve. Also includes finding the best hyperparameters for training the network.

![]( "Confusion Matrix after 50 iterations on a simple network")

### Assignment 3: Regularized MLP using different Optimizers and activations functions.
A Muti-level perceptron model is implementeed along with regularization methods(Dropout, L1, and L2) for CFAR-10 dataset.
Five different optimizers are compared for their results and convergence time (SGD, Adam, Adagrad, Adadelta, RMSprop) along with three different activation functions (ReLu, Tanh, Sigmoid).

In each case, the results were kind of counterintuitive as `SGD` and `sigmoid` performed the best. However, it might be possible that the convergence rates simply might be different. Additionally, with the network being this shallow, the benefits of the non-linearities used typically in 'Deep' networks might simply not reflect on a network of this scale. And if run long enough, `SGD` generally converges to a better minima than other optimizers like `Adam`.

### Assignment 4: CNN for CIFAR-10 with the best hyperparameters 
The assignment implements training a CNN for CIFAR-10 dataset with the best hyperparameters. Also, activations and kernels of each Conv filter are vizualized. Additionally, the working of MaxPooling or AvgPooling is compared on CIFAR-10.

### Assignment 5: Transfer learning for custom dataset of humanoid robot classification
Collected a dataset of humanoid robots and implemented the network using transfer learning for humanoid robot classification, using the two learning scenarios.
1. Fine Tuning DenseNet.
2. Using ConvNet as fixed feature extractor.

### Assignment 6: Denoising Convolutional Autoencoders 
Implemented a Convolutional Autoencoder network along with denoising Capabilities. Additionally, the learned latent space is futher used for classification task on CIFAR10 dataset. Also, latent space of CAE is used for robot classification and it performs fairly well.

### Assignment 7: LSTM and GRU manual implementation using PyTorch
Includes the basic LSTM and GRU library implementation using Pytorch.

### Assignment 8: Deep Convolutional GAN
Includes DCGAN implementation for collected robot dataset that identifies real and fake images from generated images initially, later the model learns good enough to identify the images.
