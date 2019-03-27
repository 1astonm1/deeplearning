from MachineLearningInAction.KNN.knn import knnBase
from numpy import *

def dating_class_test():
    hoRatio = 0.50  # hold out 10%
    datingDataMat, datingLabels = knnBase.file2matrix('datingTestSet2.txt')  # load data setfrom file
    normMat, ranges, minVals = knnBase.autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = knnBase.classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)

def classify_person():
    resultlist = ["not at all", "in small dose", "in large dose"]
    percentTats = float(input("precentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned pre year?"))
    icecream = float(input("liters of icecream consumed pre years?"))
    datingDataMat, datingLabels = knnBase.file2matrix('datingTestSet2.txt')  # load data setfrom file
    normMat, ranges, minVals = knnBase.autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, icecream])
    classifierResult = knnBase.classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print("you will probably like this persion:", resultlist[classifierResult - 1])


if __name__ == '__main__':
    dating_class_test()
    classify_person()