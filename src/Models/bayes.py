import numpy as np
from collections import defaultdict


class NaiveBayes:
    def __init__(self):
        self.class_probabilities = None
        self.mean = None
        self.variance = None

    def fit(self, X, y):
        self.class_probabilities = self.calculate_class_probabilities(y)
        self.mean, self.variance = self.calculate_mean_and_variance(X, y)

    def calculate_class_probabilities(self, y):
        class_counts = defaultdict(int)
        total_samples = len(y)

        for label in y:
            class_counts[label] += 1

        class_probabilities = {
            label: count / total_samples for label, count in class_counts.items()
        }
        return class_probabilities

    def calculate_mean_and_variance(self, X, y):
        unique_classes = np.unique(y)
        mean = {}
        variance = {}

        for label in unique_classes:
            class_data = X[y == label]
            mean[label] = np.mean(class_data, axis=0)
            variance[label] = np.var(class_data, axis=0)

        return mean, variance

    def gaussian_probability(self, x, mean, variance):
        exponent = np.exp(-((x - mean) ** 2) / (2 * variance))
        return (1 / (np.sqrt(2 * np.pi * variance))) * exponent

    def calculate_class_probabilities_given_features(self, features, label):
        class_probability = np.log(self.class_probabilities[label])

        for i, feature in enumerate(features):
            mean = self.mean[label][i]
            variance = self.variance[label][i]
            epsilon = 1e-10
            if variance < epsilon:
                variance = epsilon

            class_probability += np.log(
                self.gaussian_probability(feature, mean, variance)
            )

        return class_probability

    def predict(self, X):
        predictions = []

        for sample in X.values:
            class_probabilities = {
                label: self.calculate_class_probabilities_given_features(sample, label)
                for label in self.class_probabilities
            }

            predicted_class = max(class_probabilities, key=class_probabilities.get)
            predictions.append(predicted_class)

        return np.array(predictions)
