from helper import *
from PolynomialRegressor import PolynomialRegressor
import matplotlib .pyplot as plt

X, y = load_reg1dataset()
model = PolynomialRegressor(degree=1, learning_rate=0.1, max_iter=10)
for epoch in range(100):
    model.fit(X, y)
    model.plot_function(X, y,
                             titleStr="Epoch %d" % ((epoch + 1) * model.max_iter))
    
plt.ioff()
plt.show()
