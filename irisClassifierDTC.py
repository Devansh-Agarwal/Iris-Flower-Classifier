import numpy as np 
from sklearn.datasets import load_iris
from sklearn import tree

clf = tree.DecisionTreeClassifier()

#loading dataset
iris = load_iris()

#part of data set used for testing
test_idx = [107, 78 , 33]

#creating training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis = 0)

#creating testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# fitting the decision tree on data
clf = clf.fit(train_data, train_target)

print("testing \nCorrect Answer:")
print(test_target)
print("Predicted Answer:")
print(clf.predict(test_data))
