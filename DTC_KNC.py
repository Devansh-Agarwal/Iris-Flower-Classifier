from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

dtcClf = tree.DecisionTreeClassifier()
kncClf = KNeighborsClassifier()	

#loading dataset
iris = load_iris()


#creating training and testing data
train_data,test_data,train_target,test_target = train_test_split(iris.data, iris.target, test_size = .25)

#fitting training data
dtcClf = dtcClf.fit(train_data, train_target)
kncClf = kncClf.fit(train_data, train_target)

dtcPrediction = dtcClf.predict(test_data)
kncPrediction = kncClf.predict(test_data)

#checking accuracy
print("DecisionTreeClassifier Accuracy = ")
print( accuracy_score(test_target, dtcPrediction))
print("KNeighborsClassifier Accuracy = ")
print( accuracy_score(test_target, kncPrediction))
