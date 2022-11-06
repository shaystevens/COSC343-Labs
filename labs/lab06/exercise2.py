from NaiveBayesClassifier import NaiveBayesClassifier
from dataset_trec import dataset_trec
import numpy as np

trec07 = dataset_trec.load()
S = trec07.subjects
X = trec07.subject_words
y = trec07.labels

num_samples = len(X)
num_train_samples = 1000
num_test_samples = num_samples - num_train_samples
print("Training on %d samples." % num_train_samples)
print("Testing on %d samples." % num_test_samples)

random_indices = np.random. permutation(num_samples)
test_sample_indices = random_indices[:num_test_samples]
train_sample_indices = random_indices[num_test_samples:]

X = np.array(X, dtype=list)
y = np.array(y, dtype=str)
X_train = X[train_sample_indices]
y_train = y[train_sample_indices]
X_test = X[test_sample_indices]
y_test = y[test_sample_indices]

model = NaiveBayesClassifier(verbose=True)
model.fit(X_train, y_train)

y_predict = model.predict([X[0]])

model.verbose = False

y_predict = model.predict(X_test)
accuracy = np.mean(y_predict == y_test)
print("\nNaive Bayes Classifier accuracy on %d test samples: %.2f\n" % (num_test_samples, accuracy))



