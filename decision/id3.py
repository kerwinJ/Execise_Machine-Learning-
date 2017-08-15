# -*- coding: utf-8 -*-
#ID3算法是决策树的经典算法之一 
#代码参考 http://blog.csdn.net/wzmsltw/article/details/51039928
import numpy as np
import pandas as pd
from math import log
import operator

#计算数据集的信息熵
def cal_shannonEnt(dataSet):
    noEntries = len(dataSet)
    labelCounts = {}
    # 给所有可能分类创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/noEntries
        shannonEnt -= prob*log(prob, 2)
    return shannonEnt

# 对离散变量划分数据集， 取出该特征值为value的样本
def split_dataset(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
    
# 对连续变量划分数据集， direction规定划分方向，
# 决定是划分出小于value的数据样本还是大于value的数据样本集
def split_continuous_dataset(dataSet, axis, value, direction):
    retDataSet = []
    for featVec in dataSet:
        if direction == 0:
            if featVec[axis]>value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reducedFeatVec)
        else:
            if featVec[axis]<=value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reducedFeatVec)
    return retDataSet

# 选择最好的数据集划分方式
def choose_best_split(dataSet, labels):
    noFeatures = len(dataSet[0])-1
    baseEntropy = cal_shannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    bestSplitDict = {}
    for i in xrange(noFeatures):
        featList = [example[i] for example in dataSet]
        # 对连续性特征进行处理
        if type(featList[0]).__name__=='float' or type(featList[0]).__name__=='int':
            #产生n-1个候选划分点
            sortFeatList = sorted(featList)
            splitList = []
            for j in xrange(len(sortFeatList)-1):
                splitList.append((sortFeatList[j]+sortFeatList[j+1])/2.0)
            bestSplitEntropy = 10000
            slen = len(splitList)
            # 求用第j个候选划分点划分时，得到的信息熵，并记录最佳划分点
            for j in xrange(slen):
                value = splitList[j]
                newEntropy = 0.0
                subDataSet0 = split_continuous_dataset(dataSet, i, value, 0)
                subDataSet1 = split_continuous_dataset(dataSet, i, value, 1)
                prob0 = len(subDataSet0)/float(len(dataSet))
                newEntropy += prob0*cal_shannonEnt(subDataSet0)
                prob1 = len(subDataSet1)
                newEntropy += prob1*cal_shannonEnt(subDataSet1)
                if newEntropy<bestSplitEntropy:
                    beatSplitEntropy = newEntropy
                    bestSplit = j
            # 用字典记录当前特征的最佳划分点
            bestSplitDict[labels[i]] = splitList[bestSplit]
            infoGain = baseEntropy - bestSplitEntropy
        # 对离散特征进行处理
        else:
            uniqueVals = set(featList)
            newEntropy = 0.0
            # 计算该特征下每种划分的信息熵
            for value in uniqueVals:
                subDataSet = split_dataset(dataSet, i , value)
                prob = len(subDataSet)/float(len(dataSet))
                newEntropy += prob*cal_shannonEnt(subDataSet)
            infoGain = baseEntropy-newEntropy
        if infoGain>bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    #若当前节点的最佳划分特征为连续特征，则将其以之前记录的划分点为界进行二值化处理  
    #即是否小于等于bestSplitValue  
    if type(dataSet[0][bestFeature]).__name__=='float' or type(dataSet[0][bestFeature]).__name__=='int':        
        bestSplitValue=bestSplitDict[labels[bestFeature]]          
        labels[bestFeature]=labels[bestFeature]+'<='+str(bestSplitValue)  
        for i in range(shape(dataSet)[0]):  
            if dataSet[i][bestFeature]<=bestSplitValue:  
                dataSet[i][bestFeature]=1  
            else:  
                dataSet[i][bestFeature]=0  
    return bestFeature
    
#特征若已经划分完，节点下的样本还没有统一取值，则需要进行投票  
def majorityCnt(classList):  
    classCount={}  
    for vote in classList:  
        if vote not in classCount.keys():  
            classCount[vote]=0  
        classCount[vote]+=1  
    return max(classCount)
    
def createTree(dataSet, labels, data_full, labels_full):
    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat=choose_best_split(dataSet, labels)
    bestFeatLabel = labels[bestFeat]
    myTree={bestFeatLabel:{}}
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    if type(dataSet[0][bestFeat]).__name__ == 'str':
        currentLabel = labels_full.index(labels[bestFeat])
        featValuesFull = [example[currentLabel] for example in data_full]
        uniqueValsFull = set(featValuesFull)
    del(labels[bestFeat])
    # 对bestFeat的每个取值，划分出一个子树。
    for value in uniqueVals:
        subLabels=labels[:]
        if type(dataSet[0][bestFeat]).__name__ == 'str':
            uniqueValsFull.remove(value)
        myTree[bestFeatLabel][value] = createTree(split_dataset(dataSet, bestFeat, value), subLabels, data_full, labels_full)
    if type(dataSet[0][bestFeat]).__name__ == 'str':
        for value in uniqueValsFull:
            myTree[bestFeatLabel][value]=majorityCnt(classList)
    return myTree

if __name__ == '__main__':
    df = pd.read_csv('watermelon3_0_En.csv')
    data = df.values[:, 1:].tolist()
    data_full = data[:]
    labels = df.columns.values[1:-1].tolist()
    labels_full=labels[:]
    myTree = createTree(data, labels, data_full, labels_full)
    createPlot(myTree)
