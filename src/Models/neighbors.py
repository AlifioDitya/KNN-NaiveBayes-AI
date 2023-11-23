import numpy as np

class KNeighborsClassifier:
    '''
    KNeighborsClassifier is a classification algorithm that uses the k-nearest neighbors algorithm.
    It is a lazy learning algorithm that stores all instances corresponding to training data in n-dimensional space.
    When an unknown discrete data is received, it analyzes the closest k number of instances saved (nearest neighbors) and returns the most common class as the prediction and for real-valued data it returns the mean of k nearest neighbors.
    '''
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
    
    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def predict(self, X):
        X = np.array(X)
        y_pred = []
        
        for sample in X:
            distances = []
            # Calculate distance between sample and all training data
            for train_sample, train_label in zip(self.X, self.y):
                dist = self.euclidean_distance(sample, train_sample)
                distances.append((train_sample, train_label, dist))

            # Sort distances and get k nearest neighbors
            distances.sort(key=lambda x: x[2])
            neighbors = distances[:self.n_neighbors]

            # Get labels of k nearest neighbors
            labels = [neighbor[1] for neighbor in neighbors]
            unique_labels, counts = np.unique(labels, return_counts=True)

            # Get most common label
            y_pred.append(unique_labels[np.argmax(counts)])

        return y_pred
        