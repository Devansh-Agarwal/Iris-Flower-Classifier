from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
 
class bareBoneKnn():
	def fit(self, xTrain, yTrain):
		self.xTrain = xTrain;
		self.yTrain = yTrain;

	def predict(self, xTest):
		predictions = []
		for row in xTest:
			label = self.closest(row)
			predictions.append(label)
		return predictions
		
	def closest(self, row):
		shortestDist = 	distance.euclidean(self.xTrain[0], row)
		index = 0
		for i in range(1, len(self.xTrain)):
			dis = distance.euclidean(self.xTrain[i], row)
			if(shortestDist > dis ):
				index = i
				shortestDist = dis
		return self.yTrain[index]		


kncClf = bareBoneKnn()	

#loading dataset
iris = load_iris()


#creating training and testing data
train_data,test_data,train_target,test_target = train_test_split(iris.data, iris.target, test_size = .25)

#fitting training data
kncClf.fit(train_data, train_target)

kncPrediction = kncClf.predict(test_data)

#checking accuracy
print("KNeighborsClassifier Accuracy = ")
print( accuracy_score(test_target, kncPrediction))
