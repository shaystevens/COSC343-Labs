import numpy as np
import matplotlib .pyplot as plt


def hard_limiting_function(v):
    v[v <= 0] = 0
    v[v > 0] = 1
    return v


X = np.array([[-1, -1],
              [-1, 1],
              [1, -1],
              [1, 1]]).astype('float32')

y = np.array([0,
              0,
              0,
              1]).astype('uint8')

plt.scatter(X[:3, 0], X[:3, 1], c='red')
plt.scatter(X[3, 0], X[3, 1], c='blue')
plt.xlabel('x 1 ')
plt.ylabel('x 2 ')
plt.show()

N, D = np.shape(X)

K = 1

Y = np.expand_dims(y, axis=1)

max_iter = 100
learning_rate = 0.1

W = np.random.randn(D, K)
b = np.random.randn(K)

for i in range(max_iter):
    Yhat = hard_limiting_function(np.dot(X, W) + b)
    E = Y - Yhat
    deltaW = np.dot(X.T, E)
    deltab = np.sum(E, axis=0)
    W = W + learning_rate/N * deltaW
    b = b + learning_rate/N * deltab
    if np.sum(E==0):
        break


