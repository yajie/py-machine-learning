"""
    filename: load_iris_data
    author: wuyajie
    date: 2018/09/08
"""
import pandas as pd

def load_iris_data():
    """Load Iris Data"""
    iris_df = pd.read_csv('https://archive.ics.uci.edu/ml/'\
                'machine-learning-databases/iris/iris.data', header=None)
    print(iris_df.tail())
    print(iris_df.describe())

load_iris_data()
