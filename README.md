# Fashion_MNIST_Dataset_Example
This project demonstrates image classification using the Fashion-MNIST dataset with TensorFlow + Keras. Fashion MNIST resembles the classic MNIST dataset, but instead of handwritten digits, it contains images of clothing items (10 categories, such as T-shirt, trousers, shoes, and bags).

The goal of this project is to showcase how convolutional neural networks (CNNs) can classify grayscale images into fashion categories. The code is implemented in two styles: a regular procedural version and an object-oriented (OOP) version for better modularity.

# Table Of Contents
- [Implementation](#implementation)
- [Requirements](#requirments)
- [How to Use](#how-to-use)
- [Error Handling](#error-handling)
- [References](#references)

# Implementation
Dataset: Fashion MNIST - 70,000 grayscale images, 28x28 pixels, 10 classes

Model: CNN - Multiple convolutional networks and layers

Training: 10 Epochs, Adam optimizer, 0.33 validation split

The OOP code separates data loading, preprocessing, model architechture, training, and prediction into different classes. 

# Requirments 
This project requires tensorflow, keras, and scikit-learn. It was developed using a Python environment through VSCode.

Use 'pip install -r requirements.txt' to install the following dependencies:

```
tensorflow==2.20.0
keras==3.11.3
scikit-learn==1.7.1
matplotlib
numpy
```
# How to Use
To utilize this code, a Python environment is installed. Download the Fashion_MNIST.py file onto your computer into a folder. Then open that folder/file on VSCode.

# Error Handling 
This project does not have any error handling. 

# References 
- [1]GeeksforGeeks, “MNIST Dataset : Practical Applications Using Keras and PyTorch,” GeeksforGeeks, May 2024. https://www.geeksforgeeks.org/machine-learning/mnist-dataset/
‌
