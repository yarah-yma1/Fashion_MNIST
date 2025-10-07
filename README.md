# Fashion_MNIST Data Set
- This project explores the use of deep learning techniques with the Keras library to classify clothing images from the Fashion-MNIST dataset. By building and training a neural network, the goal is to enable accurate recognition and categorization of various fashion items, such as shirts, shoes, and handbags. This serves as a practical example for developing image classification models in the context of fashion and apparel.

# Table Of Contents
- [Implementation](#implementation)
- [Requirements](#requirments)
- [How to Use](#how-to-use)
- [References](#references)

# Implementation
The Fashion-MNIST dataset is a collection of 70,000 grayscale images of clothing items, such as shirts, shoes, and handbags, categorized into 10 different classes. Its purpose is to serve as a more challenging alternative to the traditional MNIST handwritten digits dataset for training and testing machine learning models. By working with this dataset, deep learning algorithms learn to recognize and classify various fashion items, enabling applications like automated product tagging and image-based search. Fashion-MNIST provides a practical benchmark for developing and evaluating image classification models in the context of clothing and fashion.

# Requirements 
- Visual Studio Code (Software)
- Python Language on Computer (3.12.0)
- GitBash (Optional)

- This project is designed to run in a VSCode terminal using a Python environment.

Use 'pip install -r requirements.txt' to install the following dependencies:
```
tensorflow==2.20.0
keras==3.11.3
scikit-learn==1.7.1
matplotlib
numpy
```

# How to Use
- To run this code, you will need to have a Python environment installed on your computer. It is recommended to use Visual Studio Code as this Python script was written and ran in VSCode. GitBash is also recommended in order to synchronize your VSCode with GitHub.
- In GitHub, click on the green icon labeled "<> CODE" on the top of this page and copy the HTTPS link.
- In VSCode, click on "Clone Git Repository" and paste the copied link from GitHub.
- In the search bar, type in "Python: Create Environment" and then select a preferred environment. This code used .venv as the virtual environment.
- When the virtual environment is open (appears as .venv in the list of items in the left menu), you may navigate to the [Class1.py](/src/Class1.py) file and select it. At this point, you may open your terminal and install the pip requirements for the necessary libraries in order to execute the code. Then, you may hit "Run" on the top right hand corner to execute the code.

- Note that this is the second iteration of the Fashion MNIST dataset which was designed to be modular, hence the classes. The original code can be seen as [FASHION.py](/src/FASHION.py)
- In addition, the Fashion_MNIST dataset originates from the KERAS library, not needing additional downloads in order to run as shown in this section of the code:
```
from keras.datasets import fashion_mnist
```

# References 
- [1]GeeksforGeeks, “Fashion MNIST with Python Keras and Deep Learning,” GeeksforGeeks, Jun. 2022. https://www.geeksforgeeks.org/deep-learning/fashion-mnist-with-python-keras-and-deep-learning/


