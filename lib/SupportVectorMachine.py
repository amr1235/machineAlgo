# Problem 1
import os
import numpy as np
from sklearn.svm import LinearSVC
import pandas as pd
from sklearn import model_selection
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

# ==================================== sklearn svm ==================== 
# path = os.path.realpath("../testData/Task_data/data.txt")
# data = pd.read_csv(path, sep=" ", header=None)
# # read the data
# X = data[data.columns[:len(data.columns) - 1]].values
# Y = data[data.columns[-1]].values
# # scaling 
# scaler = MinMaxScaler()
# scaler.fit(X)
# X = scaler.transform(X)
# # split the data
# accuracy = []
# for i in range(20):
#     X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
#         X, Y, test_size=0.4, random_state=2)
#     svm = LinearSVC(C=1, loss="hinge")
#     svm.fit(X_train, Y_train)
#     Y_pred = svm.predict(X_test)
#     accuracy.append(metrics.accuracy_score(Y_test, Y_pred))
# avg = np.average(accuracy)
# print(avg)
# ==================================== sklearn svm ====================


# build the SVM 
class SupportVectorMachine :
    def __init__(self) -> None:
        self.weights = None
    def compute_cost(self,W, X, Y,C):
        # calculate hinge loss
        N = X.shape[0]
        distances = 1 - Y * (np.dot(X, W))
        distances[distances < 0] = 0  # equivalent to max(0, distance)
        hinge_loss = C * (np.sum(distances) / N)
        # calculate cost
        cost = 1 / 2 * np.dot(W, W) + hinge_loss
        return cost

    def calculate_cost_gradient(self,W, X, Y,C):
        distance = 1 - (Y * np.dot(X, W))
        dw = np.zeros(len(W))
        for ind, d in enumerate(distance):
            if max(0, d) == 0:
                di = W
            else:
                di = W - (C * Y[ind] * X[ind])
            dw += di
        dw = dw/len(Y)  # average
        return dw
    def fit(self,X,Y,C = 1,max_iterations = 10000,learningRate = 1) :
        index = 0
        weights = np.zeros(X.shape[1])
        # gradient descent
        while(index <= max_iterations) :
            cost = self.compute_cost(weights,X,Y,C)
            weights = weights - learningRate * self.calculate_cost_gradient(weights,X,Y,C)
            newCost = self.compute_cost(weights,X,Y,C)
            if((cost - newCost) < 0) :
                learningRate = learningRate * 0.5
                weights = np.zeros(X.shape[1])
                index = 0
                continue
            if((cost - newCost) < 0.001) :
                print("minimum : " , (cost - newCost))
                break
            index += 1
        self.weights = weights
        print(weights)

    def predict(self,X) :
        return np.dot(self.weights,X)

    def accuracy_score(self,X_Test,Y_Test) :
        Y_pred = []
        for row in X_Test :
            Y_pred.append(np.sign(self.predict(row)))
        return metrics.accuracy_score(Y_Test, Y_pred)

# ==================================== usage ============================================
# from sklearn import datasets
# iris = datasets.load_iris()
# X = iris["data"][:, (2, 3)] # petal length, petal width
# Y = (iris["target"] == 2).astype(np.float64) # Iris-Virginica
# # print(X)
# X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)
# svm = SupportVectorMachine()
# svm.fit(X_train,Y_train,C = 1)
# acc = svm.accuracy_score(X_test,Y_test)
# print(acc)
# ==================================== usage ============================================

