__author__ = "Lech Szymanski"
__email__ = "lech.szymanski@otago.ac.nz"

import numpy as np
from sklearn import datasets

# AND dataset
def load_and():

    X = np.array([ [0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]]).astype('float32')
    y = np.array([ 0,
                   0,
                   0,
                   1])

    # No test data...or same test data as training data
    return (X,y)

# XOR dataset
def load_xor():
    X = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]]).astype('float32')
    y = np.array([0,
                       1,
                       1,
                       0])

    # No test data...or same test data as training data
    return X,y

def load_iris(ver2D=False):
    """ Loads the iris dataset of 4 samples.

    :param ver2D: a boolean, if True loads the D=2 attribute
                  version of the dataset, otherwise loads
                  the D=4 attribute version
    :return: (X,y) - a 150xD data matrix and corresponding 150-dim vector of labels
    """

    # Load the dataset from sklearn
    iris = datasets.load_iris()
    X = iris.data

    if ver2D:
        # If 2D mode requested use PCA to cast
        # data from 4D to 2D
        from sklearn.decomposition import PCA
        X = PCA(n_components=2, random_state=0).fit_transform(X)

    y = iris.target.astype('uint8')

    return X, y


def load_iris2D():
    """ Loads the 2D version of iris dataset.

    :return: (X,y) - a 150x2 data matrix and corresponding 150-dim vector of labels
    """
    return load_iris(ver2D=True)

# Digits dataset - 8x8 pixel image version of the MNIST dataset
def load_digits():
    digits = datasets.load_digits()

    X = digits.data
    y = digits.target

    return X, y


def load_reg1dataset():
   # Define a uniformly spaced set of x values (n in total)
   n = 150
   x = np.linspace(-5, 5, n)

   w1_true = 0.7;
   w2_true = -2;
   w3_true = 1;
   w4_true = 0.8;  # True parameters of the function
   y = w1_true * np.cos(w2_true * x) + w3_true * np.sin(w4_true * x)

   rnd = np.random.RandomState(3)

   # add some Gaussian noise to each value of the true output
   y += rnd.randn(len(y)) * 0.1
   I = rnd.permutation(len(x))
   x = x[I]
   y = y[I]

   x /= 2.5*2
   X = np.expand_dims(x, axis=1)
   y = np.expand_dims(y, axis=1)

   return (X,y)

def load_spiral():

   N = 120

   X = np.zeros((N, 2))
   y = np.zeros((N)).astype('int64')

   for i in range(0, int(N / 2)):
      delta = np.pi / 8 + i * 0.2
      r = 2 * delta + 0.1
      X[i, 0] = r * np.sin(delta)
      X[i, 1] = r * np.cos(delta)
      y[i] = 0

   for i in range(0, int(N / 2)):
      delta = np.pi / 8 + i * 0.2
      r = -2 * delta - 0.1
      X[i + int(N / 2), 0] = r * np.sin(delta)
      X[i + int(N / 2), 1] = r * np.cos(delta)
      y[i + int(N / 2)] = 1

   X -= np.min(X)
   X /= np.max(X)

   return X,y