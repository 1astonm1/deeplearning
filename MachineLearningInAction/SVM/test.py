from MachineLearningInAction.SVM.svm_base import svm

data, label = svm.loadDataSet('testSet.txt')
print(label)

b, alphas = svm.smoSimple(data, label, 0.6, 0.001, 40)
print(b)
print(alphas[alphas>0])

