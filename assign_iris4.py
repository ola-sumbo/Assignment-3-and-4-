from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import export_graphviz, ExtraTreeClassifier
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from pydot import graph_from_dot_data
import pandas as pd
from pandas import Series, DataFrame
import numpy as np

df = pd.read_csv('C:\Users\sumbo\Documents\Iris.csv')
# print df.isnull()
# print df.columns
# print df.describe()
print df.shape


# defining X as features (input) and Y as label (predicted)
x = df.iloc[:, 1:5].values
print x.shape
y = df.iloc[:, 5].values
print y.shape

# get dummies as categorical var
y = pd.get_dummies(y)
print type(y)

# evaluate the split of the training and test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# viewing shape of the new data
print X_train.shape


# for knn with k = 5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print accuracy_score(y_test, y_pred)


# with classifiers
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
print clf.predict([[3, 5, 4, 2]])
print clf.predict_proba([[3, 5, 4, 2]])


# finding the best k nearest neighbour
from sklearn import metrics

k_range = range(1, 20)
values = {}
values_list = []
for k in k_range:
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                values[k] = metrics.accuracy_score(y_test, y_pred)
                values_list.append(metrics.accuracy_score(y_test, y_pred))

# plotting the relationship btw k values and corresponding test accuracy
import matplotlib.pyplot as plt
plt.plot(k_range, values_list)
plt.xlabel('Values of K for KNN')
plt.ylabel('Test accuracy')
plt.show()  # k = 6 is the best k for our prediction

# using dictionaries to map response variable# using dictionaries to map response variable
classes = {'setsosa': 0, 'versicolor': 1,  'virginica': 2}

# instance to see different method for classification
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(x, y)

# random data set for prediction
x_new = [[3, 5, 4, 2], [2, 3, 7, 5]]
y_pred = knn.predict(x_new)
print type(y_pred)
print type(classes)

df1 = pd.DataFrame(np.vstack([y_pred]), columns=classes)
print df1








