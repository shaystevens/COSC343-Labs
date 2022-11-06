import numpy as np

from helper import *
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier

X, y = load_reg1dataset()
y = y[:, 0]

numEpochs = 100
mlpC = MLPClassifier(hidden_layer_sizes=(3, 3, 3), activation='identity', learning_rate_init=0.1, max_iter=10,
                     solver='sgd', tol=1e-5, n_iter_no_change=10000)

mlpR = MLPRegressor(hidden_layer_sizes=(6, 6, 6), activation='tanh', learning_rate_init=0.15, max_iter=1000000,
                    solver='sgd', tol=1e-5, n_iter_no_change=10000)

"""
for epoch in range(numEpochs):
    mlpC.partial_fit(X, y, np.unique(y))
    if epoch % 20 == 0:
        plot_classified_regions(mlpC, X, y, blocking=False)

plot_classified_regions(mlpC, X, y, blocking=True)
"""

for epoch in range(numEpochs):
    mlpR.partial_fit(X, y)
    plot_function(mlpR, X, y, blocking=False)

plot_function(mlpR, X, y, blocking=True)
