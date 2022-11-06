__author__ = "Lech Szymanski"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "lech.szymanski@otago.ac.nz"
__date__ = "August 2022"
__modified_by__ = "<your name>"

import numpy as np
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time


class PolynomialClassifier:
    """
           Sklearn-like model for polynomial logistic regressor trained
           with Cross-Entropy loss.

           ...

           Attributes
           ----------
           learning_rate : float
               the learning rate parameter
           max_iter : int
               maximum number of epochs to train for in the fit method
           degree: int
               degree of the polynomial
           W: numpy array
               a matrix of polynomial weights


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

           sigmoid(X):
                Static method producing sigmoid output over elements of X

           get_params()
               Returns the W matrix of the
               polynomial parameters

           set_params(**params)
               Allows setting of the W parameters of the
               polynomial parameters

           plot_classified_regions(X, y, titleStr)
               Visualisation of classified regions (works only
               if input size D=2)


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
        self.W = None

        # Set the class_labels to None - will be inferred from
        # the training data in the fit method
        self._class_labels = None

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

    @staticmethod
    def sigmoid(X):
        """
        Computes y=1/(1+e^-x) for every element x in numpy array X

        :param X: NxD numpy array of N points of D dimensions
        :
        :returns: NxD numpy array of sigmoid function applied to
                  all elements of X
        """
        return 1/(1+np.exp(-X))

    def fit(self, X, y):
        """
        Implementation of the polynomial steepest gradient descent optimisation

        :param X: numpy array
            An NxD matrix of N samples of D attributes

        :param y: numpy array
            An N-dim vector of outputs/labels

        """

        # Convert X input to Xb feature matrix
        Xb = self.input_to_poly_features(X,self.k)

        # Infer the number of training samples and
        # dimensionality of feature space input from Xb.
        N, D = Xb.shape

        # If this is the first time the fit method has
        # been invoked, initialise the model parameters.
        if self.W is None:
            # How many unique entries in y?
            self._class_labels = np.unique(y)

            if len(self._class_labels) > 2:
                # Set the number of output equal to
                # the number of unique labels, if the
                # latter is larger than 2
                K = len(self._class_labels)
            else:
                # If only 2 labels in y, one output for the model
                # is sufficient
                K = 1

            # Initialise the weight matrix with random, normally distributed
            # values
            self.W = np.random.randn(D,K)

        # Convert labels to one-hot encoding
        Y = PolynomialClassifier._one_hot(y)

        # Iterate over max_iters each time computing update based on the output
        # of the polynomial and updating the weights
        for i in range(self.max_iter):

            #Steepest gradient descent update
            #
            # You have:
            #    Xb - a NxD numpy array of inputs in feature space,
            #    Y - a NxK numpy array of desired outputs,
            #    self.W -  a DxK numpy array of weights.  
            # 
            # Must produce:
            #    Yphat - a numpy array of NxD outputs,
            # and update self.W.
            # .
            # .
            # .
            # .

            Yphat = self.sigmoid(np.dot(Xb, self.W))
            deltaw = np.dot(Xb.T, (Y*(1-Yphat) - (1-Y)*Yphat))
            self.W = self.W + self.learning_rate/N * deltaw

            if self._verbose:
                if Yphat.shape[1] == 1:
                    yhat = np.zeros(np.shape(Yphat))
                    yhat[Yphat >= 0.5] = 1
                    yhat[Yphat < 0.5] = 0
                    yhat = yhat.astype('int8')
                else:
                    yhat = np.argmax(Yphat, axis=1)

                # Return the predicted labels
                yhat = self._class_labels[yhat]

                accuracy = np.mean(yhat==y)
                print("Epoch %d/%d: accuracy %.2f" % (i + 1, self.max_iter, accuracy))

        self._trained = True

    def predict(self, X):
        """
        Computes polynomial model output Yhat = sigmoid(Xf*self.W), where Xf
        is the feature matrix from X for polynomial of degree k

        :param X: numpy array
            An NxD matrix of N samples of D attributes

        :returns: numpy array
            An N-dim vector of labels as predicted by the
            model from the input

        """

        if not self._trained:
            raise RuntimeError("Model needs training first.")

        # Convert input to features
        Xf = PolynomialClassifier.input_to_poly_features(X, self.k)

        # Calculate the weighted sum of inputs
        Yhat_prob_output = PolynomialClassifier.sigmoid(np.dot(Xf, self.W))

        # Convert probability to class labels
        if Yhat_prob_output.shape[1] == 1:
            yhat_class_indices = np.zeros(np.shape(Yhat_prob_output))
            yhat_class_indices[Yhat_prob_output >= 0.5] = 1
            yhat_class_indices[Yhat_prob_output < 0.5] = 0
            yhat_class_indices = yhat_class_indices.astype('uint8')
        else:
            yhat_class_indices = np.argmax(Yhat_prob_output,axis=1)

        # Return the predicted labels
        return self._class_labels[yhat_class_indices]

    def get_params(self):
        """
        Getter of the polynomial model parameters

        :returns: dictionary
            with polynomial weights under key 'W'.

        """

        return {'W': self.W}

    def set_params(self, **params):
        """
        Setter of the polynomial model parameters

        :param params: dictionary
            with polynomial weights under key 'W'
        """

        for key in params:
            if key != 'W':
                print("Warning! Unrecognised parameter type '%s'" % key)

        if 'W' in params:
            self.W = params['W']


    def plot_classified_regions(self, X, y, titleStr=None):
        """
        Plots classification regions for 2-attribute input

        :param X: numpy array
            Data points to plot

        :param y: numpy array
            Labels of the datapoint to plot

        :param titleStr: string
            The string to put as the title of the matplotlib plot

        """

        N, D = X.shape
        _, K = self.W.shape

        if D != 2:
            raise RuntimeError("Cannot visualise classification regions on %d-dimensional input." % D)

        if not hasattr(self, 'fh'):
            # Create a new (empty) figure
            self.fh = plt.figure()
            self.ph = self.fh.add_subplot(1,1,1)
            self.h = []
            plt.ion()
            plt.show()
            plt.xlabel('x_1')
            plt.ylabel('x_2')

        xmins = np.min(X, axis=0)
        xmaxs = np.max(X, axis=0)

        xrange = xmaxs - xmins

        xmins -= xrange * 0.2
        xmaxs += xrange * 0.2

        F = 500
        xrange1 = np.linspace(xmins[0], xmaxs[0], F)
        xrange2 = np.linspace(xmins[1], xmaxs[1], F)

        x1, x2 = np.meshgrid(xrange1, xrange2)

        x = np.concatenate([np.reshape(x1, (-1, 1)), np.reshape(x2, (-1, 1))], axis=1)

        yhat = self.predict(x)

        yc = np.zeros((len(x), 3)).astype('uint8')
        if K == 1:
            yhat = yhat[:, 0]
            yc[yhat == 0, :] = [255, 255, 255]
            yc[yhat == 1, :] = [0, 0, 255]
        elif K == 3:
            yc[yhat == self._class_labels[0], :] = [255, 0, 0]
            yc[yhat == self._class_labels[1], :] = [0, 255, 0]
            yc[yhat == self._class_labels[2], :] = [0, 0, 255]
        else:
            return

        w, h = np.shape(x1)
        yc = np.reshape(yc, (w, h, 3))
        yc = yc[::-1, :, :]

        for h in self.h:
            h.remove()

        self.h = []

        h = self.ph.imshow(yc, alpha=0.5)
        self.h.append(h)

        for i in range(len(X)):
            x = X[i]
            x0 = np.argmin(np.abs(xrange1-x[0]))
            x1 = np.argmin(np.abs(xrange2[::-1]-x[1]))

            if K == 1:
                if y[i] == self._class_labels[0]:
                    c = 'r'
                else:
                    c = 'b'
            elif K==3:
                if y[i] == self._class_labels[0]:
                    c = 'r'
                elif y[i] == self._class_labels[1]:
                    c = 'g'
                else:
                    c = 'b'

            h = plt.scatter(x0, x1, c=c, s=20)
            self.h.append(h)

        self.ph.set_xticks([])
        self.ph.set_yticks([])

        if titleStr is not None:
            plt.title(titleStr)

        plt.pause(0.1)
        time.sleep(0.1)

    @staticmethod
    def _one_hot(y):
        """
        Converts vector of labels to one-hot encoding

        :param y: numpy array
            N-dim vector of labels

        :returns: numpy array
            A NxD matrix of one hot encoded labels, where
            D is the number of unique labels in y
        """

        # Convert (N,)-shape array to (N,1)-shape
        y = np.expand_dims(y, axis=1)

        # Use Sklearn's OnHotEncoder to convert
        # digits to one-hot encoding
        enc = preprocessing.OneHotEncoder()
        enc.fit(y)
        Y = enc.transform(y).toarray()

        # If only have 2 labels, return only
        # single output
        if Y.shape[1] == 2:
            Y = Y[:, 1:]

        return Y