import numpy as np

class LinearRegression : 
    def __init__(self,parameters) -> None:
        self.parameters = parameters

    @classmethod
    def fit(cls,X,Y,numberOfIterations = 30000) :
        params = cls.gradientDescent(X,Y,numberOfIterations)
        print("parameters : ", params)
        return cls(params)

    def predict(self,X) :
        output = self.hypothesis(X,self.parameters)
        return output

    def evaluatePerformance(self,X_Test,Y_Test) :
        Y_Predicted = []
        for x in X_Test : 
            Y_Predicted.append(self.hypothesis(x,self.parameters))
        # calculate the Mean Absolute Percent Error (100 * (1/m) * sum( (actual - predicted) / actual))
        sum = 0
        for i in range(len(Y_Test)) :
            sum += abs((Y_Test[i] - Y_Predicted[i]) / Y_Test[i])
        return (1 / len(Y_Test)) * sum 
        
    @classmethod
    def gradientDescent(cls,X,Y,numberOfIterations):
        # X = [[..],[..],[..]]
        # Y = [....]
        # set initial parameters values to 0
        params = np.zeros(len(X[0]) + 1)
        reachedMinimum = False
        learningRate = 0.1
        cost = cls.computeCost(X,Y,params)
        loop = 0
        while(not reachedMinimum) :
            if loop > numberOfIterations : 
                raise Exception("Could not converge")
            params[0] = params[0] - learningRate * cls.errorSummationForW(X,Y,0,params)
            for i in range(1 , len(params)) :
                params[i] = params[i] - learningRate * cls.errorSummationForW(X,Y,i,params)  
            newCost = cls.computeCost(X,Y,params) 
            if((cost - newCost) < 0) : # check if the cost function is not converge
                learningRate = learningRate * 0.5
                params = np.zeros(len(X[0]) + 1)
                loop = 0
                continue
            elif((cost - newCost) <= 0.001) : # check if function reached Global minimum
                reachedMinimum = True
                print("minimum : " , (cost - newCost))
            elif(loop == numberOfIterations) :
                print(cost,newCost)
            else : 
                cost = newCost
            loop += 1
        return params
          
    @classmethod
    def errorSummationForW(cls,X,Y,featureNumber,params) :
        # X = [[..],[..],[..]]
        # Y = [....]
        sum = 0
        for i in range(len(X)) :
            hypo = cls.hypothesis(X[i],params)
            error = hypo - Y[i]
            if featureNumber != 0 :
                error = error * X[i][featureNumber - 1]
            sum += error
        return sum
    @classmethod
    def computeCost (cls,X,Y,params) :
        # X = [[..],[..],[..]]
        # Y = [....]
        sum = 0
        for i in range(len(X)) :
            hypo = cls.hypothesis(X[i],params)
            error = hypo - Y[i]
            errorSquare = error * error
            sum += errorSquare
        return sum * 0.5 
    
    # calculate the hypothesis function = w0 + w1X1 + w2X2 + .... + w(n)Xn 
    @staticmethod
    def hypothesis(X,params) :
        # x = [....] params = [.....]
        sum = params[0]
        for i in range(len(X)) :
            sum += params[i + 1] * X[i]
        return sum



# -------------- usage ----------------
# data = pd.read_csv("testData/multivariateData.dat",header=None)
# X_Train = data.iloc[:80,:data.shape[1] - 1].to_numpy()
# Y_Train = data.iloc[:80,data.shape[1] - 1:data.shape[1]].to_numpy()
# X_Test = data.iloc[80:,:data.shape[1] - 1].to_numpy()
# Y_Test = data.iloc[80:,data.shape[1] - 1:data.shape[1]].to_numpy()
# model = LinearRegression.fit(X_Train,Y_Train)
# pref = model.evaluatePerformance(X_Test,Y_Test)
# print("error : " , pref)
# --------------- sample output ----------
# minimum :  [0.00099686]
# parameters :  [-3.00171419  1.15173912]
# error :  4.56747470242389