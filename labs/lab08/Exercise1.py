from helper import *
from PolynomialRegressor import PolynomialRegressor
import matplotlib .pyplot as plt

X, y = load_reg1dataset()
plt.scatter(X, y, c='b')

k = 1
Xb = PolynomialRegressor.input_to_poly_features(X, degree=k)
w = np.dot(np.dot(np.linalg.inv(np.dot(Xb.T, Xb)), Xb.T), y)

xmin = np.min(X)
xmax = np.max(X)
xtest = np.linspace(xmin, xmax, 200)
Xtest = np.expand_dims(xtest, axis=1)
Xbtest = PolynomialRegressor.input_to_poly_features(Xtest, degree=k)
ytest = np.dot(Xbtest, w)
plt.plot(xtest, ytest, c='r')
plt.show()
