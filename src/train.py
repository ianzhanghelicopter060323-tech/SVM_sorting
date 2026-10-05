import numpy as np
from sklearn import svm 


def classifier() -> svm.SVC :
    """ 构建分类器函数 """
    clf = svm.SVC(
        C=0.8, # 误分类惩罚系数，控制间隔大小与训练误差之间的权衡
        kernel='linear', # 设置核函数：当前为'Linear'
        decision_function_shape='ovr' # 决策函数：计算分类分数；ovr本质上是ovo（两两比较）
    )

    return clf

def train(clf :svm.SVC, x_train: np.ndarray, y_train: np.ndarray) -> None :
    """ SVM训练函数 """
    clf.fit(x_train, y_train.ravel()) # ravel()将y_train展开成一维行向量


def train_and_evaluate(
        x_train: np.ndarray, y_train: np.ndarray,
        x_test: np.ndarray, y_test: np.ndarray, feature_name: str
        ) -> svm.SVC:
    """每组特征独立训练一个模型，使用相同参数并输出对应准确率。"""
    clf = classifier()
    train(clf, x_train, y_train)
    print('\n================ %s (%d features) ================'
          % (feature_name, x_train.shape[1]))
    print_accuracy(clf, x_train, y_train, x_test, y_test)
    return clf


def show_accuracy(y_predict: np.ndarray, y_trian: np.ndarray, tip: str) -> None :
    """ 判断a,b是否相等计算acc的均值 """
    acc = y_predict.ravel() == y_trian.ravel() # acc是一个bool型的数组
    print('%s Accuracy:%.3f' %(tip, np.mean(acc))) # mean(acc) 计算正确率

    
def print_accuracy(
        clf: svm.SVC, 
        x_train: np.ndarray, 
        y_train: np.ndarray, 
        x_test: np.ndarray, 
        y_test: np.ndarray
        ) -> None :
    """ 分别打印训练集和测试集的准确率 
        score(x_train, y_train)表示输出 x_train,y_train在模型上的准确率 
    """

    print('training prediction:%.3f' %(clf.score(x_train, y_train))) # clf.score(x_train, y_train) 模型在训练集准确率（隐式）
    print('test data prediction:%.3f' %(clf.score(x_test, y_test))) # clf.score(x_test, y_test) 模型在测试集准确率（隐式）
    
    # 原始结果和预测结果进行对比 predict() 表示对x_train样本进行预测,返回样本类别
    show_accuracy(clf.predict(x_train), y_train, 'traing data') # 训练集准确率（显式）
    show_accuracy(clf.predict(x_test), y_test, 'testing data') # 测试集准确率（显式）
    
    # ovr 返回各类别的决策分数，不是概率，也不是各分割面的几何距离
    print('decision_function:\n', clf.decision_function(x_train))
