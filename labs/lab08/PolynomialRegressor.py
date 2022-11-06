__author__ = "Lech Szymanski"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "lech.szymanski@otago.ac.nz"
__date__ = "August 2022"
__modified_by__ = "<your name>"

import numpy as np
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt
import time


class PolynomialRegressor:
    """
           Sklearn-like model for polynomial regressor trained
           with Mean Squared Error loss.

           ...

           Attributes
           ----------
           learning_rate : float
               the learning rate parameter
           max_iter : int
               maximum number of epochs to train for in the fit method
           degree: int
               degree of the polynomial
           w: numpy array
               a vector of polynomial weights


           Methods
           -------
            fit(X, y):
               Trains the polynomial hypothesis on set of points in X with
               corresponding labels in y

           predict(X)
               Returns the predictions of the polynomial for set of
               points in X

           input_to_poly_features(X,degree):
               Static method converting input matrix X to feature matrix Xf
               according to polynomial degree

           get_params()
               Returns the W matrix of the
               polynomial parameters

           set_params(**params)
               Allows setting of the W parameters of the
               polynomial parameters

           plot_function(X, y, titleStr)
               Visualisation of the model's function (works only
               if input size of D=1)


           """

    def __init__(self, degree=1, learning_rate=0.01, max_iter=100, verbose=False):
        """
        Initialisation

        :param degree: the degree of the polynomial
        :param learning_rate: the learning rate of the SGD
        :param max_iter: max number of epochs to train for in the fit method
        :param verbose: verbosity of the fit method
        """

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.k = degree
        self._verbose = verbose


        # Set the polynomial parameters to None - they will
        # be initialised to random values the first time the fit method
        # is invoked (where the number of inputs and output will be
        # inferred from the data)
        self.w = None

        self._trained = False

    @staticmethod
    def input_to_poly_features(X,degree):
        """
        Converts a NxD input matrix into a feature matrix with all combinations of
        inputs for given degree of the polynomial

        :param X: NxD numpy array of N points of D dimensions
        :param degree: the degree of the polynomial

        :returns: NxDf numpy array of X in feature space of given polynomial degree
        """

        D = np.shape(X)[1]

        X_attribute_indices = np.arange(D)
        X_features = []
        for k in range(1,degree+1):

            X_attrib_combinations = list(combinations_with_replacement(X_attribute_indices, k))
            for comb in X_attrib_combinations:
                X_f = []
                for i in comb:
                    X_select = X[:, i:i + 1]
                    X_f.append(X_select)
                X_f = np.concatenate(X_f, axis=1)
                X_f = np.expand_dims(np.prod(X_f, axis=1), axis=1)
                X_features.append(X_f)

        X_features = np.concatenate(X_features, axis=1)
        X_features = np.concatenate([X_features,np.ones((len(X_features),1))],axis=1)

        return X_features

    def fit(self, X, y):
        """
        Implementation of the polynomial steepest gradient descent optimisation

        :param X: numpy array
            An NxD matrix of N samples of D attributes

        :param y: numpy array
            An N-dim vector of outputs

        """

        # Convert X input to Xb feature matrix
        Xb = self.input_to_poly_features(X,self.k)

        # Infer the number of training samples and
        # dimensionality of feature space input from Xf.
        N, D = Xb.shape

        # If this is the first time the fit method has
        # been invoked, initialise the model parameters.
        if self.w is None:
            K = 1

            # Initialise the weight vector with random, normally distributed
            self.w = np.random.randn(D,K)

        # Iterate over max_iters each time computing update based on the output
        # of the polynomial and updating the weights
        for i in range(self.max_iter):

            #Steepest gradient descent update
            #
            # You have:
            #    Xb - a NxD numpy array of inputs in feature space,
            #    y - a Nx1 numpy array of desired outputs,
            #    self.w -  a Dx1 numpy array of weights.  
            # 
            # Must produce:
            #    yhat - a numpy array of Nx1 outputs,
            # and update self.w.
            # .
            # .
            # .
            # .
            yhat = np.dot(Xb, self.w)
            deltaw = 1/2*(np.dot(Xb.T, (y-yhat)))
            self.w = self.w + self.learning_rate/N * deltaw

            if self._verbose:
                J = np.mean(np.square(y-yhat))

                print("Epoch %d/%d: mse %.2e" % (i + 1, self.max_iter, J))

        self._trained = True

    def predict(self, X):
        """
        Computes polynomial model output yhat = Xf*self.W, where Xf
        is the feature matrix from X for polynomial of degree k

        :param X: numpy array
            An NxD matrix of N samples of D attributes

        :returns: numpy array
            An N-dim vector of outputs as predicted by the
            model from the input

        """

        if not self._trained:
            raise RuntimeError("Model needs training first.")

        # Convert input to features
        Xf = self.input_to_poly_features(X, self.k)

        # Calculate the weighted sum of inputs
        yhat = np.dot(Xf, self.w)

        # Return the output
        return yhat

    def get_params(self):
        """
        Getter of the polynomial model parameters

        :returns: dictionary
            with polynomial weights under key 'w'.

        """

        return {'w': self.w}

    def set_params(self, **params):
        """
        Setter of the polynomial model parameters

        :param params: dictionary
            with polynomial weights under key 'w'
        """

        for key in params:
            if key != 'w':
                print("Warning! Unrecognised parameter type '%s'" % key)

        if 'w' in params:
            self.w = params['w']


    def plot_function(self, X, y, titleStr=None):
        """
        Plots polynomial function for 1-attribute input

        :param X: numpy array
            Data points to plot

        :param y: numpy array
            Target values for the poly function

        :param titleStr: string
            The string to put as the title of the matplotlib plot

        """

        N, D = X.shape

        if D != 1:
            raise RuntimeError("Cannot visualise polynomial function on %d-dimensional input." % D)

        if not hasattr(self, 'fh'):
            # Create a new (empty) figure
            self.fh = plt.figure()
            self.ph = self.fh.add_subplot(1,1,1)
            self.h = []
            plt.xlabel('x')
            plt.ylabel('y')
            plt.ion()
            plt.show()

        for h in self.h:
            h.remove()

        self.h = []

        self.h.append(self.ph.scatter(X, y,c='b'))

        xmin = np.min(X)
        xmax = np.max(X)
        Xtest = np.linspace(xmin, xmax, 200)
        Xtest = np.expand_dims(Xtest, axis=1)

        yhat = self.predict(Xtest)

        h = self.ph.plot(Xtest, yhat, c='r')

        xrange = xmax-xmin
        xmin = xmin-0.1*xrange
        xmax = xmax+0.1*xrange

        self.ph.set_xlim((xmin,xmax))

        ymin = np.min(y)
        ymax = np.max(y)
        yrange = ymax-ymin
        ymin = ymin-0.1*yrange
        ymax = ymax+0.1*yrange

        self.ph.set_ylim((ymin,ymax))
        self.h.append(h[0])



        if titleStr is not None:
            plt.title(titleStr)

        plt.pause(0.1)
        time.sleep(0.1)

