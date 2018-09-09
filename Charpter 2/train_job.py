"""
    filename: train_job.py
    author: wuyajie
    date: 2018/09/09
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def show_iris_data():
    """show Iris Data"""
    iris_df = pd.read_csv('https://archive.ics.uci.edu/ml/'\
                'machine-learning-databases/iris/iris.data', header=None)
    y = iris_df.iloc[0:100, 4].values
    #Iris-setosa:1,Iris-virginica:-1
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = iris_df.iloc[0:100, [0, 2]].values
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versiclolor')
    plt.xlabel('petal length')
    plt.ylabel('sepal length')
    plt.legend(loc='upper left')
    plt.show()



