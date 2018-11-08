#!/Users/coffeexc/anaconda2/envs/L3/bin/python
# -*- coding:utf-8 -*-

from numpy import *
import os

# 创建测试数据
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# k-近邻算法
def classify0(inX, dataSet, labels, k):

    # 距离计算
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # 计算每个点和测试点每个维度之间的距离
    sqDiffMat = diffMat ** 2  # 计算每个点和测试点每个维度之间距离的平方
    sqDistance = sqDiffMat.sum(axis=1)  # 求平方和
    distances = sqDistance ** 0.5  # 求 L2范数 矩阵

    # 将 k-近邻 内的标签放入字典中并计数
    sortedDistIndicies = distances.argsort()  # 通过距离排序，排序的结果是原索引
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        if voteIlabel in classCount:
            classCount[voteIlabel] += 1
        else:
            classCount[voteIlabel] = 1

    # 将结果标签字典排序
    sortedClassCount = sorted(classCount.items(), reverse=True)

    return sortedClassCount[0][0]

# ***********************************
# 用 k-近邻算法改进约会网站的配对效果
# 数据：stSet2.txt
# ***********************************

# 将文本记录转化为Numpy
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)  # 得到文件行数

    # 创建返回的Numpy矩阵
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0

    #
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index += 1

    return returnMat, classLabelVector


# 归一化特征值
def autoNorm(dataSet):

    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


# 分类器针对约会网站的测试代码
def datingClassTest():

    hoRatio = 0.10  # 用于测试的数据比例
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)

    errorCount = 0.0  # 错误率

    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 2)
        print("the classifier came back with: %s, the real answer is : %s" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total true rate is: %f" % (1 - errorCount / float(numTestVecs)))


# ***********************************
# 手写识别系统
# 数据：digits/
# ***********************************

# 将手写数字图片转化为向量
def img2vector(filename):

    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


# 手写数字识别系统测试代码
def handwritingClassTest():

    # 获取目录内容
    hwLabels = []
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        # 从文件名中解析分类数字
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])

        # 构造训练集
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('digits/traingDigits/%s' % fileNameStr)

    testFileList = os.listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print("\nthe total number of errors is : %d" % errorCount)
    print("\nthe total true rate is: %f" % (1 - errorCount / float(mTest)))


if __name__ == "__main__":

    print("hello World")