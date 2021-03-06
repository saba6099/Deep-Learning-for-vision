

# Vision Systems Lab: Learning Computer Vision on GPUs using Deep Learning Models
Implemented as a part of the CudaVision Lab at University of Bonn (Winter Semester 2019).
Code repository contains various deep learning models implemented using PyTorch, and Python as Jupyter notebooks. 

Contributors: Saba Khan, [Tejas Morbagal Harish](https://github.com/TejasMorbagal/)

### Assignment 1: Logistic Regression Classifier using pure python/numpy
Contains a 2 layer network implementation for Logistic Regression Classifier using pure python/numpy, using softmax activation function and mean squared error. The network is trained on MNIST data.

### Assignment 2: Logistic Regression Classifier dataset using PyTorch
Contains the implementation of a Logistic Regression Classifier with CIFAR-10 dataset using PyTorch, along with the confusion matrix and the learning curve. Also includes finding the best hyperparameters for training the network.

### Assignment 3: Regularized MLP using different Optimizers and activations functions.
A Muti-level perceptron model is implementeed along with regularization methods(Dropout, L1, and L2) for CIFAR-10 dataset.
Five different optimizers are compared for their results and convergence time (SGD, Adam, Adagrad, Adadelta, RMSprop) along with three different activation functions (ReLu, Tanh, Sigmoid). Figures show the training loss for different activation functions.

![](https://github.com/saba6099/Deep-Learning-for-vision/blob/master/Assignment%203/relu.png)
![](https://github.com/saba6099/Deep-Learning-for-vision/blob/master/Assignment%203/sigmoid.png)
![](https://github.com/saba6099/Deep-Learning-for-vision/blob/master/Assignment%203/tan.png)

### Assignment 4: CNN for CIFAR-10 with the best hyperparameters 
The assignment implements training a CNN for CIFAR-10 dataset with the best hyperparameters. Also, kernels of each Conv filter are visualized. Additionally, the working of MaxPooling or AvgPooling is compared on CIFAR-10 dataset.

![](https://github.com/saba6099/Deep-Learning-for-vision/blob/master/Assignment%204/kernel_visualization.png "Kernel Visualizations")

### Assignment 5: Transfer learning for custom dataset of humanoid robot classification
Collected a dataset of humanoid robots and implemented the network using transfer learning for humanoid robot classification, using the two learning scenarios.
1. Fine Tuning DenseNet.
2. Using ConvNet as fixed feature extractor.

![Humanoid robot classification results](https://github.com/saba6099/Deep-Learning-for-vision/blob/master/Assignment%205/result1.png )

### Assignment 6: Denoising Convolutional Autoencoders 
Implemented a Convolutional Autoencoder network along with denoising Capabilities. Additionally, the learned latent space is futher used for classification task on CIFAR10 dataset. Also, latent space of CAE is used for robot classification and it performs fairly well.

!["Actual Images"](https://github.com/saba6099/Deep-Learning-for-vision/blob/master/Assignment%206/actual.png)
![Reconstructed images](https://github.com/saba6099/Deep-Learning-for-vision/blob/master/Assignment%206/reconstructed.png)

### Assignment 7: LSTM and GRU manual implementation using PyTorch
Includes the basic LSTM and GRU library implementation using Pytorch.

### Assignment 8: Deep Convolutional GAN
Includes DCGAN implementation for collected robot dataset.

![](https://github.com/saba6099/Deep-Learning-for-vision/blob/master/Assignment%208/real_images.png "Actual Images")
![](https://github.com/saba6099/Deep-Learning-for-vision/blob/master/Assignment%208/generated_images.png "Generated Images")

