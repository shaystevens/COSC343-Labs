from helper import *
from PolynomialClassifier import PolynomialClassifier
import matplotlib .pyplot as plt

# X, y = load_xor()
# X, y = load_iris2D()
X, y = load_spiral()

model = PolynomialClassifier(degree=1, learning_rate=0.1, max_iter=10)

for epoch in range(100):
    model.fit(X, y)
    model.plot_classified_regions(X, y, titleStr="Epoch %d" % ((epoch + 1)*model.max_iter))

plt.ioff()
plt.show()
