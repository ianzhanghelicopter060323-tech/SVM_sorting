import numpy as np
from matplotlib import colors
from sklearn import svm 
from sklearn import model_selection


def classifier() -> svm.SVC :
    """ 构建分类器函数 """
    clf = svm.SVC(
        C=0.8, # 误差项惩罚系数：对于某次“错误”，修正力度有多大
        kernel='linear', # 设置核函数：当前为'Linear'
        decision_function_shape='ovr' # 决策函数：计算分类分数；ovr本质上是ovo（两两比较）
    )

    return clf

def train(clf :svm.SVC, x_train: np.ndarray, y_train: np.ndarray) :
    """ SVM训练函数 """
    clf.fit(x_train, y_train.ravel()) # ravel()将y_train展开成一维行向量

