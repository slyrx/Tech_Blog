---
layout: post
title:  "scikit learn: RBF SVM parameters"
date:   2018-07-21 15:08:30
tags: [机器学习, 数据挖掘, scikit-learn, Support Vector Machines]
---

    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import load_iris
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.model_selection import GridSearchCV

    class MidpointNormalize(Normalize): # 执行正规化类的

        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            Normalize.__init__(self, vmin, vmax, clip) # Normalize正规化函数，将对应的处理值处理到0和1之间

        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y)) # interp表示插值函数，就是说通过x,y求得一个函数，再把value带入这个函数，得出的是针对这个函数value对应的y值。masked_array检查其中的元素是否为真。此处的value是一个257的数组，通过1维线性函数可以得出对应的257个y值

    iris = load_iris()
    X = iris.data
    y = iris.target

    X_2d = X[:, :2]
    X_2d = X_2d[y > 0] # 取对应类大于0的点
    y_2d = y[y > 0]
    y_2d -= 1

    scaler = StandardScaler() # 通过删除均值和缩放到单位方差来标准化特征
    X = scaler.fit_transform(X)
    X_2d = scaler.fit_transform(X_2d)

    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv) # 在最佳参数的选择上，是通过输入参数param_grid和cv来作为域进行搜索判断的，param_grid是一组值，而cv是交叉验证用到的数据集。
    grid.fit(X, y) # 在众多备选值中找到了一组最优值，用这一组最优做拟合，求出模型

    print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

    C_2d_range = [1e-2, 1, 1e2]
    gamma_2d_range = [1e-1, 1, 1e1]
    classifiers = []
    for C in C_2d_range:
        for gamma in gamma_2d_range:
            clf = SVC(C=C, gamma=gamma)
            clf.fit(X_2d, y_2d)
            classifiers.append((C, gamma, clf)) # 形成一系列的分类器，这里是一组，不一定是最优的

    plt.figure(figsize=(8, 6))
    xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    for (k, (C, gamma, clf)) in enumerate(classifiers):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
        plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)), size='medium')

        plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r, edgecolors='k')
        plt.xticks(())
        plt.yticks(())
        plt.axis('tight')

    scores = grid.cv_results_['mean_test_score'].reshape(len(C_range), len(gamma_range))


    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot, norm=MidpointNormalize(vmin=0.2, midpoint=0.92)) # norm参数表示对数据进行正规化，数据范围调整到(0，1)之间
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar() # 把当前fig画布中的像素块中涉及到的颜色汇总成一个颜色条。
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.show()

end
