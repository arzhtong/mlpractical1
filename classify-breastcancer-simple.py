# classify-iris-simple.py
# parsons/29-jan-2017
#
# Using a decision tree on the iris dataset.
#
# Code is based on:
#
# http://scikit-learn.org/stable/modules/tree.html)
# http://scikit-learn.org/stable/auto_examples/tree/plot_iris.html

import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_breast_cancer
# Parameters
plot_step = 0.02

# Load the data

# X = bc.data  # 1 and 3 are the features we will use here.
# y = bc.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
#
# # Now create a decision tree and fit it to the iris data:
# cancer_tree = tree.DecisionTreeClassifier(criterion="entropy").fit(X_train, y_train)

# Now plot the decision surface that we just learnt by using the decision tree to
# classify every packground point.
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
#                      np.arange(y_min, y_max, plot_step))

# Z = cancer_tree.predict(np.c_[xx.ravel(), yy.ravel()]) # Here we use the tree
                                                     # to predict the classification
                                                     # of each background point.
# Z = Z.reshape(xx.shape)
# cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
#
# # Also plot the original data on the same axes
# plt.scatter(X[:, 0], X[:, 1], c=y.astype(np.float))
#
# # Label axes
# plt.xlabel( bc.feature_names[1], fontsize=10 )
# plt.ylabel( bc.feature_names[3], fontsize=10 )
#
# plt.show()
list=[]
bc = load_breast_cancer()
X = bc.data  # 1 and 3 are the features we will use here.
y = bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10)
value=0
sd=0
# Now create a decision tree and fit it to the iris data:
for i in range(10):
    cancer_tree = tree.DecisionTreeClassifier(criterion="entropy").fit(X_train, y_train)
    z=cancer_tree.score(X_test, y_test)
    list.append(z)
for item in list:
    value+=item
value=value/len(list)
print (value)
print("Standard Error" + )
