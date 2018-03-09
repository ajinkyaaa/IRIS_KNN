

import numpy as np


from sklearn import datasets
from scipy.spatial import distance

def eucledean(a,b):
        return distance.euclidean(a,b)


iris = datasets.load_iris()


iris_X = iris.data

iris_y = iris.target

np.unique(iris_y)


np.random.seed(1)


indices = np.random.permutation(len(iris_X))

iris_X_train = iris_X[indices[:105]]
iris_y_train = iris_y[indices[:105]]
iris_X_test  = iris_X[indices[105:]]
iris_y_test  = iris_y[indices[105:]]




from collections import Counter
class myCustomClassifier():
    def __init__(self,n_number = 3):
        self.n_number = n_number
    
    def fit(self,iris_X_train,iris_y_train):
        self.iris_X_train = iris_X_train
        self.iris_y_train = iris_y_train
        
    def closest(self,row):
        
        tempDist = []
        tempFull = []
       
        counter = 0
        
        for i in range(1,len(iris_X_train)):
            
            dist = eucledean(row,self.iris_X_train[i])
            tempDist.append((dist,self.iris_y_train[i]))
           
        
        tempFull = [i[1] for i in sorted(tempDist)[:self.n_number]]
        voteResult = Counter(tempFull).most_common(1)[0][0]
     
        #Take vote of k number of closest train data and return one with most vote      
        return voteResult
    
    
        
    def predict(self,x_test):
        
        predictions = []
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions


from sklearn.neighbors import KNeighborsClassifier

#Input k nearest neighbour
knn = myCustomClassifier(3)
knn.fit(iris_X_train, iris_y_train) 

Final_predictions = knn.predict(iris_X_test)


from sklearn.metrics import accuracy_score
print("Accuracy :- ",accuracy_score(iris_y_test,Final_predictions))




