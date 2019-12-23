import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split

# creating decision tree
features = ['raining']
X = [[0], [1]]
Y = [0, 1]

clf = tree.DecisionTreeClassifier()
clf.fit(X, Y)

# visualising the tree
trefile = open("dtree.dot", 'w')
tree.export_graphviz(clf, out_file=trefile, feature_names=features, filled=True, rounded=True, impurity=False, class_names=['No Umbrella', 'Umbrella'])
trefile.close()


print clf.predict([[0]])









