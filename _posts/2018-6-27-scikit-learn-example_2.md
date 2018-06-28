---
layout: post
title:  "scikit learn: Classifier comparison"
date:   2018-06-27 21:05:30
tags: [机器学习, 数据挖掘, scikit-learn]
---

    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors import ListedColormap
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_moons, make_circles, make_classification
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticSiscriminantAnalysis

    h = .02

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost", "Naive Bayes", "QDA"]

    classifier = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 \* RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    X,y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1) # 生成随机多类问题需要的多数据；n_features特征的总数；n_informative提供有用信息的特征;n_redundant冗余特征；random_state随机状态，由随机数生成一个种子，随机种子；n_clusters_per_class每个聚类类别的个数

    rng = np.random.RandomState(2)# 随机种子生成
    X += 2 * rng.uniform(size=X.shape) # 从一个均匀分布[low，high)中随机采样，注意定义域是左闭右开，即包含low，不包含high
    linearly_separable = (X,y) #X,y合并，生成一个数组的tuple

    datasets = [make_moons(noise=0.3, random_state=0), # make_moons生成两个插入的半圆形
                make_circles(noise=0.2, factor=0.5, random_state=1), #生成一个大圆套着小圆，共两个圆
                linearly_separable] # 将半圆数据、圆形数据、生成tuple三种数据组合成一个list，

    figure = plt.figure(figsize=(27,9)) # 生成一张画布，竖轴方向每单元80像素，横轴方向每单元53像素
    i = 1

    for ds_cnt, ds in enumerate(datasets):
        X,y = ds
        X = StandardScaler().fit_transform(X) #通过消除到单位方差的均值和缩放，来标准化特征
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.4, random_state=42)# 做数据切片，分割出训练集和测试集

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5 # 取x的最大值、最小值，并做放大和缩小
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5 # 取y的最大值、最小值，并做放大和缩小
        xx,yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) #生成多维矩阵

        cm = plt.cm.RdBu # 红蓝组合的图谱，颜色推荐 https://matplotlib.org/examples/color/colormaps_reference.html
        cm_bright = ListedColormap(['#FF0000', '#0000FF']) #自定义过渡颜色图谱
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i) #子图大小及位置设置
        if ds_cnt == 0:
            ax.set_title("Input data") # 输入数据图片部分设置标题

        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k") # 训练集数据绘制散点图
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k') # 同一画布上，测试集数据也绘制散点图
        ax.set_xlim(xx.min(), xx.max()) # 横轴设置显示范围
        ax.set_ylim(yy.min(), yy.max()) # 纵轴设置显示范围
        ax.set_xticks(()) # 去掉横轴的刻度
        ax.set_yticks(()) # 去掉纵轴的刻度
        i += 1

        for name, clf in zip(names, classifiers):# 依次使用数组中的分类器
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            clf.fit(X_train, y_train) # 拟合训练集的数据
            score = clf.score(X_test, y_test) # 对测试集的结果打分

            if hasattr(clf, "decision_function"):# 返回这个clf对象是否含有“decision_function”属性
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            Z = Z.reshape(xx,shape) # 对预测结果进行整形，恢复原有的对应关系
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8) # 图像的轮廓

            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name)
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')
            i += 1

    plt.tight_layout() # 提供关于调整子图们更好的适应画布的基本操作routines
    plt.show()




### 分类器比较
#### 代码概述
+ 大量的代码其实是用做绘图可视化的
+ 用到模型方法的地方只有，fit和predict
+ 前面的 **数据整形** 和后面的 **绘图调整** 占据了代码的99%的篇幅

#### 主要的分类器使用步骤
+ clf.***fit***(X_train, y_train)
+ score = clf.***score***(X_test, y_test)
+ Z = clf.***predict_proba***(np.c_[xx.ravel(), yy.ravel()])[:, 1]
+ Z = clf..***decision_function.***(np.c_[xx.ravel(), yy.ravel()])
+ .***predict_proba*** 和 ***decision_function.*** 不会同时出现

### 数据形式
+ 拟合使用的数据维度(60,2)
+ 打分使用的数据维度(40,2)
+ 预测值使用的数据维度xx是(272,248)，yy也是(272，248)
> 预测值变化过程：
> + (272,248)压缩成1维的，xx,yy各压缩一次
> + 将两个1维的xx，yy拼接成2维的，即(:,2),2列的所有行,总行数等于272*248=67456
> + 模型方法对(:,2)预测，预测结果是(:,1),67456的单列值
> + 将预测结果还原成(272，248)的输入形态，通过图形化方法表示出来，进行观察























end
