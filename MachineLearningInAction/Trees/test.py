from MachineLearningInAction.Trees.trees import trees_base
from MachineLearningInAction.Trees.trees import trees_plotter

mytree = trees_plotter.retrieveTree(0)
trees_plotter.createPlot(mytree)

mytree['no surfacing'][3]='maybe'
print(mytree)
trees_plotter.createPlot(mytree)