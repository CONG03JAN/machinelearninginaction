#!/Users/coffeexc/anaconda2/envs/L3/bin/python
# -*- coding:utf-8 -*-

# Logistic 回归
# 优点：计算代价不高，易于理解和实现
# 缺点：容易欠拟合，分类精度可能不高
# 适用数据类型：数值型和标称型数据


import numpy as np

# ***********************
# 使用梯度上升找到最佳参数
#
# 每个回归系数初始化为 1
# 重复 R 次：
#   计算整个数据集的梯度
#   使用 alpha * gradient 更新回归系数的向量
# 返回回归系数
# ***********************


# 获取训练数据
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append((int(lineArr[2])))
    return dataMat, labelMat


# 构造 Sigmooid 函数
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


# 梯度上升算法
def gradAscent(dataMatIn, classLabels):

    dataMatrix = np.mat(dataMatIn)  # 转化为 Numpy 矩阵
    labelMat = np.mat(classLabels).transpose()  # 转换为 Numpy 矩阵并转置
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


# ***********************
# 随机梯度上升
#
# 每个回归系数初始化为 1
# 对数据集中每个样本：
#   计算整个数据集的梯度
#   使用 alpha * gradient 更新回归系数的向量
# 返回回归系数
# ***********************


# 随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * np.array(dataMatrix[i]) * error
    return weights


# 改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classlabels, numIter=150):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classlabels[randIndex] - h
            weights = weights + alpha * error * np.array(dataMatrix[randIndex])
            del(dataIndex[randIndex])
    return weights


# Logistic 回归分类函数
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1
    else:
        return 0
    

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if float(classifyVector(np.array(lineArr), trainWeights)) != float(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount / numTestVec))
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iteratiions the average error rate is: %f" % (numTests, errorSum / float(numTests)))
    np.array()


if __name__ == "__main__":

    # dataMat, labelMat = loadDataSet()
    # print(gradAscent(dataMat, labelMat))
    # print(stocGradAscent0(dataMat, labelMat))
    # print(stocGradAscent1(dataMat, labelMat))
    multiTest()
