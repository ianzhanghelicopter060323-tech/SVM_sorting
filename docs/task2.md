使用支持向量机（SVM）实现鸢尾花分类
对于SVM，存在一个分类面，两个点集到此平面的最小距离最大，两个点集中的边缘点到此平面的距离最大。 

本实践使用SVM算法实现鸢尾花的分类

加载相关包
In [1]
import numpy as np
from matplotlib import colors
from sklearn import svm 
from sklearn import model_selection
import matplotlib.pyplot as plt
import matplotlib as mpl
加载数据、切分数据集
In [2]
# ======将字符串转化为整形==============
def iris_type(s):
    it = {b'Iris-setosa':0, b'Iris-versicolor':1,b'Iris-virginica':2} 
    return it[s]
    
# 1 数据准备
# 1.1 加载数据
data = np.loadtxt('/home/aistudio/data/data2301/iris.data',  # 数据文件路径i
                  dtype=float,    # 数据类型
                  delimiter=',',  # 数据分割符
                  converters={4:iris_type}) # 将第五列使用函数iris_type进行转换
# 1.2 数据分割
x, y = np.split(data, (4, ), axis=1) # 数据分组 第五列开始往后为y 代表纵向分割按列分割
x = x[:, :2]
x_train, x_test, y_train, y_test=model_selection.train_test_split(x, y, random_state=1, test_size=0.2)
构建SVM分类器，训练函数
In [3]
# SVM分类器构建
def classifier():
    clf = svm.SVC(C=0.8,                         # 误差项惩罚系数
                  kernel='linear',               # 线性核 高斯核 rbf
                  decision_function_shape='ovr') # 决策函数
    return clf
    
# 训练模型
def train(clf, x_train, y_train):
    clf.fit(x_train, y_train.ravel()) # 训练集特征向量和 训练集目标值
    
初始化分类器实例，训练模型
In [4]
# 2 定义模型 SVM模型定义
clf = classifier()
# 3 训练模型
train(clf, x_train, y_train)
展示训练结果及验证结果
In [5]
# ======判断a,b是否相等计算acc的均值
def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print('%s Accuracy:%.3f' %(tip, np.mean(acc)))
    
# 分别打印训练集和测试集的准确率 score(x_train, y_train)表示输出 x_train,y_train在模型上的准确率
def print_accuracy(clf, x_train, y_train, x_test, y_test):
    print('training prediction:%.3f' %(clf.score(x_train, y_train)))
    print('test data prediction:%.3f' %(clf.score(x_test, y_test)))
    # 原始结果和预测结果进行对比 predict() 表示对x_train样本进行预测,返回样本类别
    show_accuracy(clf.predict(x_train), y_train, 'traing data')
    show_accuracy(clf.predict(x_test), y_test, 'testing data')
    # 计算决策函数的值 表示x到各个分割平面的距离
    print('decision_function:\n', clf.decision_function(x_train))
    
def draw(clf, x):   
    iris_feature = 'sepal length', 'sepal width', 'petal length', 'petal width'
    # 开始画图 
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    # 生成网格采样点
    x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  
    # 测试点
    grid_test = np.stack((x1.flat, x2.flat), axis = 1)
    print('grid_test:\n', grid_test)
    # 输出样本到决策面的距离
    z = clf.decision_function(grid_test)
    print('the distance to decision plane:\n', z)
    grid_hat = clf.predict(grid_test)
    # 预测分类值 得到[0, 0, ..., 2, 2]
    print('grid_hat:\n', grid_hat)
    # 使得grid_hat 和 x1 形状一致
    grid_hat = grid_hat.reshape(x1.shape)
    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'b', 'r'])
    
    plt.pcolormesh(x1, x2, grid_hat, cmap = cm_light) 
    plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), edgecolor='k', s=50, cmap=cm_dark )
    plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolor='none', zorder=10 )
    plt.xlabel(iris_feature[0], fontsize=20) 
    plt.ylabel(iris_feature[1], fontsize=20)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('Iris data classification via SVM', fontsize=30)
    plt.grid()
    plt.show()

# 4 模型评估
print('-------- eval ----------')
print_accuracy(clf, x_train, y_train, x_test, y_test)
# 5 模型使用
print('-------- show ----------')
draw(clf, x) 