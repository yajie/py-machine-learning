"""
    filename: perception.py
    author: wuyajie
    date: 2018/09/08
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Perception:
    """Perception classifier.
    Parameters
    ------------
    eta : float
        Learning rate, between 0.0 and 1.0.
    n_iter : int
        Passes over the training datase. 
    Attributes
    ------------
    weight : 1d-array
        Weights after fitting
    errors_ : list
        Number of misclassifications in every epoch.
    """

    def __init__(self, eta=0.01, n_iter=10):
        """init perception"""
        self.eta = eta
        self.n_iter = n_iter
        self.weight = []
        self.errors_ = []

    def fit(self, x_train, y_target):
        """Fit training data.
        Parameters
        ------------
        x_train : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y_target : array-like, shape = [n_samples]
            Target values.

        Returns
        """
        self.weight = np.zeros(1 + x_train.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for x_train_i, target in zip(x_train, y_target):
                update = self.eta * (target - self.predict(x_train_i))
                self.weight[1:] += update * x_train_i
                self.weight[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, x_train_):
        """Calculate net input
        np.dot is used to calculates (z = W' * X)
        """
        return np.dot(x_train_, self.weight[1:]) + self.weight[0]

    def predict(self, x_train_):
        """Return class label after unit step"""
        return np.where(self.net_input(x_train_) >= 0.0, 1, -1)

    def hello(self):
        """hello"""
        print("hello perception")

    def train_job(self):
        """train job"""
        iris_df = pd.read_csv('https://archive.ics.uci.edu/ml/'\
                    'machine-learning-databases/iris/iris.data', header=None)
        y = iris_df.iloc[0:100, 4].values
        #Iris-setosa:1,Iris-virginica:-1
        y = np.where(y == 'Iris-setosa', -1, 1)
        X = iris_df.iloc[0:100, [0, 2]].values
        #ppn = Perception(eta=0.1, n_iter=10)
        self.fit(X, y)
        print(self.errors_)
        print(self.weight)
        plt.plot(range(1, len(self.errors_) + 1), self.errors_, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Number of misclassifications')
        plt.show()

#show_iris_data()
Perception(eta=0.1, n_iter=10).train_job()
