import numpy as np
import operator
import matplotlib.pyplot as plt


def creatDataSet():
    group = np.array([1, 1.1], [1, 1], [0, 0], [0, 0.1])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, lables, k):
    dataSetSize = dataSet.shape[0]  # 只输出行数


if __name__ == '__main__':
    group, lables = creatDataSet()




