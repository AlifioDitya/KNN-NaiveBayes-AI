import numpy as np

class ID3Classifier:
    '''
    ID3 (Iterative Dichotomiser 3) is an algorithm used to generate a decision tree from a dataset.
    It is a greedy algorithm that selects the best attribute to split the data based on information gain.

    ID3Classifier requires a dataset with encoded categorical features and a categorical target variable.
    '''
    def __init__(self):
        self.tree = None

    def fit(self, X, y, verbose=False):
        self.X = X
        self.y = y
        self.tree = self._build_tree(np.array(X), np.array(y), verbose)

    def predict(self, X):
        return [self._traverse_tree(x, self.tree) for x in X]

    def _build_tree(self, X, y, verbose=False):
        # Base cases: if all samples have the same class or there are no features left
        if len(np.unique(y)) == 1:
            return y[0] # Return the class
        if len(X[0]) == 0:
            return np.bincount(y).argmax() # Return the most common class

        # Find the best feature and threshold to split the data
        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None or best_threshold is None:
            # No split found that reduces entropy (information gain = 0)
            return np.bincount(y).argmax()

        # Split the data based on the best feature and threshold
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        if verbose:
            feature_name = self.X.columns[best_feature]
            print("Splitting on feature", feature_name, "with threshold", best_threshold)
            print()

        # Recursively build the left and right subtree
        left_tree = self._build_tree(X[left_indices], y[left_indices], verbose)
        right_tree = self._build_tree(X[right_indices], y[right_indices], verbose)

        return {'feature': best_feature, 'threshold': best_threshold,
                'left': left_tree, 'right': right_tree}

    def _find_best_split(self, X, y):
        # How it works:
        # 1. For each feature, find all possible thresholds to split the data
        # 2. For each threshold, calculate the information gain
        # 3. Return the feature and threshold that results in the highest information gain

        # How finding the thresholds works:
        # 1. Sort the unique values of the feature
        # 2. Find the midpoint between each pair of values
        # 3. These midpoints are the possible thresholds

        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            unique_values = np.unique(X[:, feature])
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, X, y, feature, threshold):
        # How it works:
        # 1. Calculate the parent entropy
        # 2. Calculate the child entropy for the left and right splits
        # 3. Calculate the weighted average of the child entropy
        # 4. Calculate the information gain by subtracting the child entropy from the parent entropy
        # The idea is that the parent entropy should be higher than the child entropy to reduce disorder in the dataset
        # The higher the information gain, the better the split

        parent_entropy = self._entropy(y)

        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold

        # If either split is empty, skip this split
        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return 0

        left_entropy = self._entropy(y[left_indices])
        right_entropy = self._entropy(y[right_indices])

        left_weight = np.sum(left_indices) / len(y)
        right_weight = np.sum(right_indices) / len(y)

        child_entropy = left_weight * left_entropy + right_weight * right_entropy
        information_gain = parent_entropy - child_entropy

        return information_gain

    def _entropy(self, y):
        # Entropy is the order/disorder of a system
        # In machine learning, entropy is the amount of information in a dataset
        # It is calculated as the sum of the probability of each class multiplied by the log of that same probability
        # The most ordered system has the lowest entropy (0)
        # Example: a dataset with only one class has an entropy of 0
        # The most disordered system has the highest entropy (1)
        # Example: a dataset with two classes with equal probability has an entropy of 1

        # Formula: entropy = -sum(p_i * log(p_i))
        # p_i = probability of class i
        # log = logarithm base 2

        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy

    def _traverse_tree(self, x, tree):
        # Recursively traverse the tree until a leaf node is reached
        if isinstance(tree, dict):
            feature = tree['feature']
            threshold = tree['threshold']

            if x[feature] <= threshold:
                return self._traverse_tree(x, tree['left'])
            else:
                return self._traverse_tree(x, tree['right'])
        else:
            return tree
        
    def print_tree(self, tree=None, indent=''):
        if not tree:
            tree = self.tree

        if isinstance(tree, dict):
            feature_name = self.X.columns[tree['feature']]
            print(indent + 'if feature ' + str(feature_name) + ' <= ' + str(tree['threshold']) + ':')
            self.print_tree(tree['left'], indent + '  ')
            print(indent + 'else:')
            self.print_tree(tree['right'], indent + '  ')
        else:
            print(indent + 'return ' + str(tree))

class Node:
    def __init__(self, feature, threshold):
        self.feature = feature
        self.threshold = threshold
        self.left = None
        self.right = None
        self.leaf = False
        self.prediction = None

class DecisionTreeClassifier:
    '''
    DecisionTreeClassifier is a decision tree classifier that uses the CART (Classification and Regression Tree) algorithm.
    It is a greedy algorithm that selects the best attribute to split the data based on the Gini impurity.
    '''
    def __init__(self, max_depth=5):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _best_split(self, X, y):
        m = y.size
        if m <= 1:
            return None, None

        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None

        for idx in range(self.n_features_):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                )
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(feature=None, threshold=None)
        if depth < self.max_depth and len(num_samples_per_class) > 1:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
                return node
        node.leaf = True
        node.prediction = predicted_class
        return node
    
    def _predict(self, inputs):
        node = self.tree_
        while not node.leaf:
            if inputs[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction