import numpy as np

class LogisticRegression:
    '''
    Logistic regression is a classification algorithm used to assign observations to a discrete set of classes.
    This implementation assumes a task of binary classification (2 classes).
    '''
    def __init__(self, learning_rate=0.01, epoch=1000):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Initialize weights and bias
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.epoch):
            # Linear combination and activation function (sigmoid)
            linear_comb = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_comb)

            # Update weights and bias with gradient descent
            # Using binary cross entropy loss function, only for binary classification
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_comb = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_comb)
        y_pred_class = [1 if prob >= 0.5 else 0 for prob in y_pred]
        return y_pred_class