import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

iris = load_iris()
print(iris.keys())

# exploratory on data
# print(iris.data)
print iris.data.shape
print iris.target.shape

# load data-sets
df = pd.DataFrame(iris.data)  # load the dataset as a pandas data frame
y = iris.target            # define the target variable (dependent variable) as y

# create training and testing vars
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)
print X_train.shape, y_train.shape
print X_test.shape, y_test.shape

# fit a model
import sklearn
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model = lm.fit(X_train, y_train)
predict_test = lm.predict(X_test)
predict_train = lm.predict(X_train)

# predict from train test
print predict_train[0:5]

## The line / model
plt.scatter(y_test, predict_test)
plt.title('Accuracy plot of actual and predicted x')
plt.ylabel('prediction')
plt.xlabel('actual values')
plt.show()

# accuracy check
print model.score(X_test, y_test)

# cross validation
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics

# Perform 6-fold cross validation
scores = cross_val_score(model, df, y, cv=6)
print scores

# Make cross validated predictions
predictions = cross_val_predict(model, df, y, cv=6)
plt.scatter(y, predictions)
plt.show()

accuracy = metrics.r2_score(y, predictions)
print accuracy








