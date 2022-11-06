from helper import *
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn import *

X, y = load_mnist()
'''
X = np.reshape(X, (70000, 784))

X_train = X[:60000]
X_test = X[60000:]
y_train = y[:60000]
y_test = y[60000:]

numEpochs = 100
mlpC = MLPClassifier(hidden_layer_sizes=(3, 3, 3), activation='identity', learning_rate_init=0.1, max_iter=10,
                     solver='sgd', tol=1e-5, n_iter_no_change=10000)

for epoch in range(numEpochs):
    mlpC.partial_fit(X, y, np.unique(y))
    if epoch % 20 == 0:
        plot_classified_regions(mlpC, X, y, blocking=False)

plot_classified_regions(mlpC, X, y, blocking=True)

test_accuracy = mlpC.score(X_test, y_test)
print("Test accuracy: %f" % (test_accuracy))
'''
