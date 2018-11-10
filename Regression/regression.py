#!/Users/coffeexc/anaconda2/envs/L3/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from scipy import linalg



# 数据导入
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


# 标准线性回归
def standRegres(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T

    xTx = xMat.T * xMat

    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return

    ws = xTx.I * (xMat.T * yMat)

    return ws


# 局部加权线性回归
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))

    for j in range(m):
        # 权重大小以指数级衰减
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))

    xTx = xMat.T * (weights * xMat)

    if linalg.det(xTx) == 0.0:
        print("this matrix is singular, cannot do inverse")
        return

    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


# 岭回归
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print("this matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = ws.T
    return wMat


# 前向逐步回归
# eps 迭代步长
# numIt 迭代步数
def rssError(A, B):
    return sum((A - B) ** 2)


def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0); yMat = yMat - yMean
    xMean = np.mean(xMat, 0); xVar = np.var(xMat, 0); xMat = (xMat - xMean) / xVar

    m, n = np.shape(xMat)

    returnMat = np.zeros((numIt, n))

    ws = np.zeros((n, 1)); wsTest = ws.copy(); wsMax = ws.copy()

    for i in range(numIt):
        print("numIt is: " + str(i) + " ", ws.T)
        lowestError = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T

    return returnMat


if __name__ == '__main__':

   dataMat, labelMat = loadDataSet("abalone.txt")
   ws = stageWise(dataMat, labelMat, 0.001, 5000)
   print(ws)


