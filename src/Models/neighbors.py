import numpy as np
import pandas as pd
import pickle
import concurrent.futures
from os import cpu_count
from tqdm import tqdm
import time

class KNN:
    '''
    KNN is a classification algorithm that uses the k-nearest neighbors algorithm.
    It is a lazy learning algorithm that stores all instances corresponding to training data in n-dimensional space.
    When an unknown discrete data is received, it analyzes the closest k number of instances saved (nearest neighbors) and returns the most common class as the prediction and for real-valued data it returns the mean of k nearest neighbors.
    '''
    def __init__(self, k=3, n_jobs=1, verbose=True):
        # Constructor
        self.k = k
        self.verbose = verbose

        # If n_jobs is -1, use all available cores
        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs

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
        
    def _predict_instance(self, row):
        # Predict a single instance
        neighbours = self.get_nearest_neighbours(row)
        labels = [self.y_train.iloc[neighbour] for neighbour in neighbours]
        return max(set(labels), key=labels.count)

    def predict(self, X_test):
        if self.verbose:
            print(f"Using {self.n_jobs} {'core' if self.n_jobs == 1 else 'cores'} for predictions.")

        if isinstance(X_test, pd.DataFrame):
            if X_test.columns.empty:
                X_test = X_test.values.astype(float)
            else:
                X_test = X_test.iloc[:, :-1].values.astype(float)
        else:
            X_test = X_test.astype(float)

        start_time = time.time()

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            results = list(tqdm(executor.map(self._predict_instance, X_test), total=len(X_test)))

        elapsed_time = time.time() - start_time

        if self.verbose:
            print(f"Prediction completed in {elapsed_time:.2f} seconds.")

        return np.array(results)
    
    def save(self, path):
        pickle.dump(self, open(path, 'wb'))
