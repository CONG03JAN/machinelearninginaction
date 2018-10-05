# k近邻算法 - KD树

import numpy as np

# 创建训练数据集
def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 构造KD树
def createKDTree(dataSet, depth=0):
    n = np.shape(dataSet)[0]
    treeNode = {}
    if n == 0:
        return None
    else:
        n, m = np.shape(dataSet)
        split_axis = depth % m
        depth += 1
        treeNode['split'] = split_axis
        dataSet = sorted(dataSet, key=lambda a: a[split_axis])
        num = n // 2
        treeNode['median'] = dataSet[num]
        treeNode['left'] = createKDTree(dataSet[:num], depth)
        treeNode['right'] = createKDTree(dataSet[num + 1:], depth)
        return treeNode


# 搜索KD树
def searchTree(tree, data):
    k = len(data)
    if tree is None:
        return [0] * k, float('inf')
    split_axis = tree['split']
    median_point = tree['median']
    if data[split_axis] <= median_point[split_axis]:
        nearestPoint, nearestDistance = searchTree(tree['left'], data)
    else:
        nearestPoint, nearestDistance = searchTree(tree['right'], data)
    nowDistance = np.linalg.norm(data - median_point)
    if nowDistance < nearestDistance:
        nearestDistance = nowDistance
        nearestPoint = median_point.copy()
    splitDistance = abs(
        data[split_axis] - median_point[split_axis])
    if splitDistance > nearestDistance:
        return nearestPoint, nearestDistance
    else:
        if data[split_axis] <= median_point[split_axis]:
            nextTree = tree['right']
        else:
            nextTree = tree['left']
        nearPoint, nearDistanc = searchTree(nextTree, data)
        if nearDistanc < nearestDistance:
            nearestDistance = nearDistanc
            nearestPoint = nearPoint.copy()
        return nearestPoint, nearestDistance
