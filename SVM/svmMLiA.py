#!/Users/coffeexc/anaconda2/envs/L3/bin/python
# -*- coding:utf-8 -*-

# SVM应用的一般框架
# 1. 收集数据：可以使用任意方法
# 2. 准备数据：需要数值型数据
# 3. 分析数据：有助于可视化分割超平面
# 4. 训练算法：SVM的大部分时间都源自训练，该过程主要实现两个参数的调优
# 5. 测试算法：十分简单地计算过程就可以实现
# 6. 使用算法：几乎所有分类问题都可以使用SVM，值得一提的是，SVM本身是一个二分类分类器，对多分类问题应用SVM需要对代码做一些修改

import random
import numpy as np


# ***************
# 简化版SMO算法
# ***************


# SMO算法辅助函数

# 读取数据
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


# 随机选择 0 ~ m 中一个不等于 i 的整数
# i 为 alpha 的下标，m 为 alpha 的个数
def selectJrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j

# 调整大于 H 或者小于 L 的 alpha 值
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


# Simple SMO
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = np.mat(dataMatIn); labelMat = np.mat(classLabels).transpose()
    b = 0; m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))
    numIter = 0
    while numIter < maxIter:
        alphaPairsChanged = 0  # 用于记录 alpha 是否已经进行优化
        for i in range(m):
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])
            if (labelMat[i] < -toler and alphas[i] < C) or (labelMat[i] * Ei > toler and alphas[i] > 0):
                j = selectJrand(i, m)
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # 保证 alpha 在 0 ~ C 之间
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L == H")
                    continue

                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                      dataMatrix[i, :] *dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T

                if eta >= 0:
                    print("eta >= 0")
                    continue

                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print("j not moving enough")
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])



if __name__ == '__main__':

    dataArr, labelArr = loadDataSet('testSet.txt')
