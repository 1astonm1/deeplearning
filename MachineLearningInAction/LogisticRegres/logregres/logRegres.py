from numpy import *
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []
    lableMat = []
    fr = open('testSet.txt')
    for line in fr.readline():
        lineArr = line.strip().split()
        #print(lineArr)
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        lableMat.append(int(lineArr[2]))
    return dataMat, lableMat


def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def grandAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatrix)

