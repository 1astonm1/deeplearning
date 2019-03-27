from MachineLearningInAction.Trees.trees import trees_base
from MachineLearningInAction.Trees.trees import trees_plotter

myDat, labels = trees_base.createDataSet()
mytree = trees_plotter.retrieveTree(0)
output = trees_base.classify(mytree, labels, [1, 0])
print(output)
output =trees_base.classify(mytree, labels, [1, 1])
print(output)