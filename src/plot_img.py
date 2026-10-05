import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from matplotlib.lines import Line2D
from sklearn import svm


def draw(clf: svm.SVC, x: np.ndarray, y: np.ndarray, x_test: np.ndarray) -> None :   
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
    plt.legend(
        handles=[
            Line2D([0], [0], marker='o', color='w', label='Iris-setosa',
                   markerfacecolor='g', markeredgecolor='k', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Iris-versicolor',
                   markerfacecolor='b', markeredgecolor='k', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Iris-virginica',
                   markerfacecolor='r', markeredgecolor='k', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Test sample',
                   markerfacecolor='none', markeredgecolor='k', markersize=10),
        ],
        loc='best',
    )
    plt.grid()
    plt.savefig("/home/ianzhang/ai_foundation_prog/SVM_sorting/pics/SVM_sorting.jpg")
    plt.show()
