import numpy as np
import pandas as pd

class KNN:
    '''
    KNN is a classification algorithm that uses the k-nearest neighbors algorithm.
    It is a lazy learning algorithm that stores all instances corresponding to training data in n-dimensional space.
    When an unknown discrete data is received, it analyzes the closest k number of instances saved (nearest neighbors) and returns the most common class as the prediction and for real-valued data it returns the mean of k nearest neighbors.
    '''
    def __init__(self, k = 3):
        # Constructor
        self.k = k

    def euclidean_distance(self, row_1, row_2):
        # Calculate euclidean distance between two rows
        row_1 = row_1.astype(float)
        row_2 = row_2.astype(float)
        return np.sqrt(np.sum((row_1 - row_2) ** 2))
    
    def change_k(self, k):
        # Change k number
        self.k = k

    def get_nearest_neighbours(self, test):
        # Get nearest neighbours
        distances = np.sqrt(np.sum((self.X_train - test) ** 2, axis=1))
        indices = np.argsort(distances)[:self.k]
        return indices
    
    def fit(self, X_train, y_train):
        # Save train data
        if isinstance(X_train, pd.DataFrame):
            if X_train.columns.empty:
                self.X_train = X_train.values.astype(float)
            else:
                self.X_train = X_train.iloc[:, :-1].values.astype(float)
        else:
            self.X_train = X_train.astype(float)

        self.y_train = y_train
        
    def predict(self, X_test):
        # Predict data from train
        y_pred = []

        if isinstance(X_test, pd.DataFrame):
            if X_test.columns.empty:
                X_test = X_test.values.astype(float)
            else:
                X_test = X_test.iloc[:, :-1].values.astype(float)
        else:
            X_test = X_test.astype(float)

        for row in X_test:
            neighbours = self.get_nearest_neighbours(row)
            labels = [self.y_train.iloc[neighbour] for neighbour in neighbours]
            prediction = max(set(labels), key=labels.count)
            y_pred.append(prediction)
        
        return np.array(y_pred)
        