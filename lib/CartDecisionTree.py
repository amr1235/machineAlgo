import pandas as pd
class CartDecisionTree:
    class Node:
        def __init__(self,featuer = None,trueValue = None,depth = None,gini = None , entropy = None,isLeaf = False,cls = None):
            self.yes = None
            self.no = None
            self.label = featuer
            self.trueValue = trueValue
            self.gini= gini
            self.entropy = entropy
            self.isLeaf = isLeaf
            self.cls = cls
            self.depth = depth

    def __init__(self, Data,depth=10):  # data is a dataframe
        self.data = Data
        # self.depth = 10
        self.featuers = Data.columns[:len(Data.columns) - 1]
        self.root = None
        self.depth = depth

    def getYesAndNoDataSets(self,table,feature,value) :

        yesDataSet = table[table[feature] == value]
        yesDataSet = table.drop(columns=[feature])
        noDataSet = table[table[feature] != value]
        return (yesDataSet,noDataSet)

    def getYesAndNoDataSets2(self,table,feature,value) : 
        yesDataSet = table[table[feature] == value]
        yesDataSet = yesDataSet.drop(columns=[feature])
        noDataSet = table[table[feature] != value]
        return (yesDataSet,noDataSet)

    def claculateImpurity(self,table, featuer):
        uniqueValues = table[featuer].unique()
        targets = table[table.columns[-1]].unique()
        # calculate for each value in the featuer number of Instances in each class
        # Outlook              NumberOfInstances   No  Yes
        # Sunny                5                   3.0  2.0
        # Overcast             4                   0.0  4.0
        # Rain                 5                   2.0  3.0
        result = pd.DataFrame(columns=[featuer, "NumberOfInstances"])
        for target in targets:
            result[target] = []
        for value in uniqueValues:
            NumberOfInstances = table[table[featuer] == value].values.__len__()
            row = [value, NumberOfInstances]
            for target in targets:
                counts = table[(table[featuer] == value) & (
                    table[table.columns[-1]] == target)].values.__len__()
                row.append(counts)
            result.loc[len(result.index)] = row
        # calculate gini indeces
        (giniTotal,minimumValue) = self.giniIndex(result,featuer)
        return (giniTotal,minimumValue)
    def GetAvgClass(self,table) :
        List = table[table.columns[-1]].values.tolist()
        return max(set(List), key = List.count)
    def train(self):
        yesStack = []
        noStack = []
        nodesStack = []
        totalDepth = self.depth
        # get root 
        root = self.getNextNode(self.data) 
        self.root = self.Node(featuer = root["name"],trueValue=root["minimumValue"],depth = 0,gini = root["giniTotal"])
        (yesDataSet,noDataSet) = self.getYesAndNoDataSets(self.data,root["name"],root["minimumValue"])
        nodesStack.append(self.root)
        yesStack.append(yesDataSet)
        noStack.append(noDataSet)
        print("root : " + root["name"] + " Gini : " + str(root["giniTotal"]) + " true : " + str(root["minimumValue"]))
        # print(self.root.depth)
        # print(noDataSet)
        # depth = 1
        # test = nodesStack.pop()
        # test.no = self.Node("test","test",0.5)
        # print(self.root.no.label)
        # print(len(nodesStack))
        while(len(nodesStack) != 0) :
            # print("entered")
            skipRight = False
            skipLeft = False
            node = nodesStack.pop()
            # reset depth
            yesData = yesStack.pop()
            noData = noStack.pop()
            (yesYesDataSet,yesNoDataSet) = (None,None)
            (noYesDataSet,noNoDataSet) = (None,None)
            if(not node) : 
                continue
            if(node.isLeaf) : 
                continue
            depth = node.depth + 1
            print(depth)
            if(len(yesData.columns) == 1) : 
                cls = self.GetAvgClass(yesData)
                node.yes = self.Node(isLeaf=True,cls=cls)
                skipRight = True
                print("right of this node : " + str(node.label) + " with gini : " + str(node.gini) + " is a leaf node of class : " + str(cls))
            if(len(noData.columns) == 1) :
                cls = self.GetAvgClass(noData)
                node.yes = self.Node(isLeaf=True,cls=cls)
                skipLeft = True
                print("left of this node : " + str(node.label) + " with gini : " + str(node.gini) +" is a leaf node of class : " + str(cls))
            if((not skipRight) and (not yesData.empty)) :
                yesNode = self.getNextNode(yesData)
                node.yes = self.Node(featuer=yesNode["name"],trueValue=yesNode["minimumValue"],depth=depth,gini=yesNode["giniTotal"])
                (yesYesDataSet,yesNoDataSet) = self.getYesAndNoDataSets(yesData,yesNode["name"],yesNode["minimumValue"])
                print("true of " + node.label + " name : " + yesNode["name"] + " Gini : " + str(yesNode["giniTotal"]) + " true : " + str(yesNode["minimumValue"]))
            if((not skipLeft) and (not noData.empty)) : 
                noNode = self.getNextNode(noData)
                node.no = self.Node(featuer=noNode["name"],trueValue=noNode["minimumValue"],depth=depth,gini = noNode["giniTotal"])
                print("no of " + node.label + " name : " + noNode["name"] + " Gini : " + str(noNode["giniTotal"]) + " true : " + str(noNode["minimumValue"]))
                (noYesDataSet,noNoDataSet) = self.getYesAndNoDataSets(noData,noNode["name"],noNode["minimumValue"])

            if(depth != totalDepth) :
                nodesStack.append(node.no)
                nodesStack.append(node.yes)

                noStack.append(noNoDataSet)
                yesStack.append(noYesDataSet)

                noStack.append(yesNoDataSet)
                yesStack.append(yesYesDataSet)
            if(depth == totalDepth) : 
                # check if the node not already asigned as a leaf 
                if(node.yes) : 
                    if (not node.yes.isLeaf) : 
                        # create leaf node with the avg data
                        if(not yesYesDataSet.empty) : 
                            yesCls = self.GetAvgClass(yesYesDataSet)
                            node.yes.yes = self.Node(isLeaf=True,cls=yesCls)
                            print("right of this node : " + str(node.yes.label) + " with gini : " + str(node.yes.gini) + " is a leaf node of class : " + str(yesCls))
                        if(not yesNoDataSet.empty) :
                            noCls = self.GetAvgClass(yesNoDataSet)
                            node.yes.no = self.Node(isLeaf=True,cls=noCls)
                            print("left of this node : " + str(node.yes.label) + " with gini : " + str(node.yes.gini) + " is a leaf node of class : " + str(noCls))
                if(node.no) :
                    if(not node.no.isLeaf):
                        if(not noYesDataSet.empty) : 
                            yesCls = self.GetAvgClass(noYesDataSet)
                            print("right of this node : " + str(node.no.label) + " with gini : " + str(node.no.gini) + " is a leaf node of class : " + str(yesCls))
                            node.no.yes = self.Node(isLeaf=True,cls=yesCls)
                        if(not noNoDataSet.empty) :
                            noCls = self.GetAvgClass(noNoDataSet)
                            node.no.no = self.Node(isLeaf=True,cls=noCls)
                            print("left of this node : " + str(node.no.label) + " with gini : " + str(node.no.gini) + " is a leaf node of class : " + str(noCls))
        # print(self.root.no.no.no.label,self.root.no.no.no.trueValue,self.root.no.no.no.gini)
        # print(self.root.yes.yes.yes.yes.label,self.root.yes.yes.yes.trueValue,self.root.yes.yes.yes.gini)
        # print(self.root.yes.label,self.root.yes.trueValue,self.root.yes.gini)
        print("done")

    def giniIndex(self,table,featuer) :
        # this function calculates the gini index for each value and calculate the gini for the whole featuer
        # it also returns the value of the smallest gini index in that featuer
        # gini(featuer = value) = 1 - (table[target] / table["NumberOfInstances"])^2
        targets = table.columns[2:]
        # print(table)
        TotalNumberOfInstances = table["NumberOfInstances"].sum()
        giniTotal = 0
        ginis = {}
        # print(table,targets)
        for i in range(table.__len__()) : 
            gini = 1
            for target in targets : 
                gini += -1 * (table.iloc[i][target] / table.iloc[i]["NumberOfInstances"]) ** 2
            ginis[table.iloc[i][featuer]] = gini
            giniTotal += gini * (table.iloc[i]["NumberOfInstances"]/ TotalNumberOfInstances)
        minimumValue = min(ginis,key=ginis.get)
        return (giniTotal,minimumValue)
    def predict(self,row) :
        # prepair data to predict 
        # Sunny	Hot	High Weak
        dataDict = {}
        for i in range(len(self.featuers)) :
            dataDict[self.featuers[i]] = row[i]
        currentNode = self.root
        while(not currentNode.isLeaf) :
            compare = currentNode.label
            compareTo = currentNode.trueValue
            if(dataDict[compare] == compareTo) : # true
                currentNode = currentNode.yes
            else : 
                currentNode = currentNode.no
        return currentNode.cls
    def getAccuracy(self,X_Test,Y_Test) :
        numberOfTrues = 0
        index = 0
        for row in X_Test : 
            prediction = self.predict(row)
            if(prediction == Y_Test[index]) :
                numberOfTrues += 1
            index += 1
        print("number Of trues is : " + str(numberOfTrues) + " ; Total Number is " + str(len(Y_Test)))
        return (numberOfTrues / len(Y_Test)) * 100



    def getNextNode(self,table):
        featuers = table.columns[:len(table.columns) - 1]
        minimum = None
        for featuer in featuers:
            featureInfo = {}
            (giniTotal,minimumValue) = self.claculateImpurity(table, featuer)
            featureInfo["name"] = featuer
            featureInfo["giniTotal"] = giniTotal
            featureInfo["minimumValue"] = minimumValue
            if(not minimum) :
                minimum = featureInfo
            else : 
                if(featureInfo["giniTotal"] < minimum["giniTotal"]) : 
                    minimum = featureInfo
        return minimum
# ==================================== usage ==================================
# from sklearn import tree
# from sklearn import model_selection
# import numpy as np
# import os

# path = os.path.realpath("../testData/cardio_train.csv")

# data = pd.read_csv(path,header=0, sep=";")
# header = ["gender","ap_hi","ap_lo","cholesterol","gluc","smoke","alco","active","cardio"]
# data = data[header]
# X = data[data.columns[0:len(data.columns) - 1]]
# X = X.values.tolist()
# Y = data["cardio"].tolist()
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.1)
# data = pd.DataFrame(np.array(X_train),columns=header[:len(header) - 1])
# data["cardio"] = y_train

# tree = CartDecisionTree(data,depth=5)
# tree.train()
# tree.getAccuracy(X_test,y_test)
# =============================================================================