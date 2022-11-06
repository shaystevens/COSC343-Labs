import numpy as np


class NaiveBayesClassifier():

    def __init__(self, verbose=False):
        self.prob_data_given_lbl = None
        self.priors = None
        self.class_labels = None
        self.verbose = verbose
        self.trained = False

    def fit(self,X,y):
        self.class_labels, class_counts = np.unique(y, return_counts=True)
        self.class_labels = self.class_labels.tolist()
        n_classes = len(self.class_labels)
        self.priors = class_counts / np.sum(class_counts)
        if self.verbose:
            print("Found %d class labels: %s" %
                  (n_classes, str(self.class_labels )))
            print("Priors p(c):")
            for k in range(n_classes):
                class_label = self.class_labels[k]
                prob = self.priors[k]
                print(" P(c=’%s ’)=%.2f" % (class_label, prob))

        self.prob_data_given_lbl = []
        for _ in self.class_labels:
            self.prob_data_given_lbl.append(dict())
        for sample_index, data_sample in enumerate(X):
            sample_label = y[sample_index]
            k = self.class_labels.index(sample_label)
            for data_val in data_sample:
                if data_val in self.prob_data_given_lbl[k]:
                    self.prob_data_given_lbl[k][data_val] += 1
                else:
                    self.prob_data_given_lbl[k][data_val] = 1
        for k in range(len(self.class_labels)):
            for data_val in self.prob_data_given_lbl[k]:
                self.prob_data_given_lbl[k][data_val] /= class_counts[k]
        self.trained = True

    def predict(self,X):
        if not self.trained:
            raise RuntimeError("Model needs training first.")

        num_samples = len(X)
        num_classes = len(self.class_labels)
        predictions = []

        for sample_index in range(num_samples):
            data_sample = X[sample_index]
            lkhood_class_given_data = np.copy(self.priors)
            if self.verbose:
                print("Prediction for sample %d" % sample_index)
                print(" Data: %s" % str(data_sample))
                print(" Priors: %s" % str(lkhood_class_given_data))

            for data_val in data_sample:
                for k in range(num_classes):
                    if data_val in self.prob_data_given_lbl[k]:
                        prob = self.prob_data_given_lbl[k][data_val]
                    else:
                        prob = 0
                    lkhood_class_given_data[k] *= prob
                if self.verbose:
                    print(" After ’%s’:" % str(data_val))
                    print(" Lkhood: %s" % str(lkhood_class_given_data))

            k = np.argmax(lkhood_class_given_data)
            class_label = self.class_labels[k]
            predictions.append(class_label)
            if self.verbose:
                print(" Predct: %s" % class_label)
                
        return predictions
