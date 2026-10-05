from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from sklearn import svm
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def draw(clf: svm.SVC, x: np.ndarray, y: np.ndarray, x_test: np.ndarray) -> None :   
    """绘制原萼片模型的二维决策区域。"""
    _draw_2d(clf, x, y, x_test, ('sepal length', 'sepal width'),
             'Iris SVM: sepal features', 'SVM_sorting.jpg')


def draw_petal(clf: svm.SVC, x: np.ndarray, y: np.ndarray, x_test: np.ndarray) -> None:
    """绘制花瓣模型的二维决策区域，坐标轴对应花瓣特征。"""
    _draw_2d(clf, x, y, x_test, ('petal length', 'petal width'),
             'Iris SVM: petal features', 'SVM_petal.jpg')


def _save_and_show(fig, filename: str) -> None:
    """按项目位置保存，先保存再显示，关闭后释放画布。"""
    output_dir = Path(__file__).resolve().parent.parent / 'pics'
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def _draw_2d(clf, x, y, x_test, iris_feature, title, filename) -> None:
    """两个二维模型共用绘图逻辑，背景与散点使用一致的类别颜色。"""
    fig = plt.figure(figsize=(8, 6), constrained_layout=True)
    # 开始画图 
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    # 留出边距，避免极值样本和测试集圆圈被坐标边界裁切。
    x1_margin = (x1_max - x1_min) * 0.05
    x2_margin = (x2_max - x2_min) * 0.05
    x1_min, x1_max = x1_min - x1_margin, x1_max + x1_margin
    x2_min, x2_max = x2_min - x2_margin, x2_max + x2_margin
    # 生成网格采样点
    x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  
    # 测试点
    grid_test = np.stack((x1.flat, x2.flat), axis = 1)
    grid_hat = clf.predict(grid_test)
    # 预测分类值 得到[0, 0, ..., 2, 2]
    # 使得grid_hat 和 x1 形状一致
    grid_hat = grid_hat.reshape(x1.shape)
    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#A0A0FF', '#FFA0A0'])
    cm_dark = mpl.colors.ListedColormap(['g', 'b', 'r'])
    
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light, vmin=0, vmax=2, shading='auto')
    plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), edgecolor='k', s=50,
                cmap=cm_dark, vmin=0, vmax=2)
    plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolor='none', edgecolor='k', zorder=10)
    plt.xlabel(iris_feature[0] + ' (cm)', fontsize=14)
    plt.ylabel(iris_feature[1] + ' (cm)', fontsize=14)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(title + '\nAll samples; circles mark test samples', fontsize=15)
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
    _save_and_show(fig, filename)


def draw_all_features(clf: svm.SVC, x_test: np.ndarray, y_test: np.ndarray) -> None:
    """展示四特征模型的测试结果：二维投影和混淆矩阵。

    预测时传入完整四列；花瓣坐标仅用于展示样本位置。
    四维决策区域不能直接画在二维平面上，因此不绘制决策背景。
    """
    y_true = y_test.ravel()
    y_predict = clf.predict(x_test)
    wrong = y_predict != y_true
    accuracy = np.mean(~wrong)
    class_names = ['setosa', 'versicolor', 'virginica']
    cm_dark = mpl.colors.ListedColormap(['g', 'b', 'r'])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, labels, title in zip(axes[:2], [y_true, y_predict],
                                 ['True labels', 'Predicted labels (4 features)']):
        ax.scatter(x_test[:, 2], x_test[:, 3], c=labels, cmap=cm_dark,
                   vmin=0, vmax=2, edgecolor='k', s=65)
        ax.scatter(x_test[wrong, 2], x_test[wrong, 3], marker='x',
                   color='k', s=130, linewidths=2, label='Misclassified')
        ax.set_xlabel('petal length (cm)')
        ax.set_ylabel('petal width (cm)')
        ax.set_title(title)
        ax.grid(alpha=0.3)
    handles = [Line2D([0], [0], marker='o', linestyle='none', color=color,
                      label=name) for color, name in zip(['g', 'b', 'r'], class_names)]
    handles.append(Line2D([0], [0], marker='x', linestyle='none', color='k',
                          label='Misclassified'))
    axes[0].legend(handles=handles, fontsize=9)

    matrix = confusion_matrix(y_true, y_predict, labels=[0, 1, 2])
    ConfusionMatrixDisplay(matrix, display_labels=class_names).plot(
        ax=axes[2], cmap='Blues', colorbar=False, values_format='d')
    axes[2].set_title('Test confusion matrix')
    fig.suptitle('Iris SVM: all 4 features | Test accuracy: %.2f%% (%d/%d)\n'
                 'Scatter plots: petal-coordinate projections of test samples'
                 % (100 * accuracy, np.sum(~wrong), len(y_true)), fontsize=14)
    _save_and_show(fig, 'SVM_all_features.jpg')
