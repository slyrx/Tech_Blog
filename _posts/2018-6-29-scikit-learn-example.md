---
layout: post
title:  "scikit learn: Linear and Quadratic Discriminant Analysis with covariance ellipsoid"
date:   2018-06-29 09:26:30
tags: [机器学习, 数据挖掘, scikit-learn]
---

    print(__doc__)

    from scipy import linalg
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import colors

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    cmap = colors.LinearSegmentedColormap("red_blue_classes",{'red':[(0,1,1), (1, 0.7, 0.7)], 'green':[(0, 0.7, 0.7), (1, 0.7, 0.7)], 'blue':[(0, 0.7, 0.7), (1, 1, 1)]}) #自定义配色类，用于显示
    plt.cm.register_cmap(cmap = cmap)

    def dataset_fixed_cov():
        n, dim = 300, 2
        np.random.seed(0)
        C = np.array([[0., -0.23], [0.83, .23]]) # 固定的2x2矩阵
        X = np.r_[np.dot(np.random.randn(n, dim), C), # np.dot二维矩阵计算乘积，同线性代数中矩阵乘法的定义；对于一维矩阵，计算两者的内积。
         np.dot(np.random.randn(n, dim), C) + np.array([1,1])] # np.r_表示按列连接两个矩阵，就是把两个矩阵上下拼接，要求列数相等；np.c_表示按行连接两个矩阵，就是把两个矩阵左右拼接，要求行数相等。np.r_相当于pandas中的concat(),np.c_相当于pandas中的merge()
        y = np.hstack((np.zeros(n), np.ones(n))) # np.hstack表示左右拼接两个矩阵; np.zeros(n)生成一个list，里面有n个0; np.ones(n)生成一个list，里面有n个1
        return X,y

    def dataset_cov():
        n, dim = 300, 2
        np.random.seed(0)
        C = np.array([[0., -1.], [2.5, .7]]) * 2.
        X = np.r_[np.dot(np.random.randn(n, dim), C), np.dot(np.random.randn(n, dim), C.T) + np.array([1,4])]
        y = np.hstack((np.zeros(n), np.ones(n)))
        return X,y

    def plot_data(lda, X, y, y_pred, fig_index):
        splot = plt.subplot(2,2,fig_index)
        if fig_index == 1:
          plt.title('Linear Discriminant Analysis')
          plt.ylabel('Data with\n fixed covariance')
        elif fig_index == 2:
          plt.title('Quadratic Discriminant Analysis')
        elif fig_index == 3:
          plt.ylabel('Data with\n varying covariances')

        tp = (y == y_pred)
        tp0, tp1 = tp[y == 0], tp[y == 1]
        X0, X1 = X[y == 0], X[y == 1]
        X0_tp, X0_fp = X0[tp0], X0[~tp0]
        X1_tp, X1_fp = X1[tp1], X1[~tp1]

        alpha = 0.5

        plt.plot(X0_tp[:, 0], X0_tp[:, 1], 'o', alpha=alpha, color='red', markeredgecolor='k')
        plt.plot(X0_fp[:, 0], X0_fp[:, 1], '\*', alpha=alpha, color='#990000', markeredgecolor='k') # 标号代表含义参考 https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot

        plt.plot(X1_tp[:, 0], X1_tp[:, 1], 'o', alpha=alpha, color="blue", markeredgecolor='k')
        plt.plot(X1_fp[:, 0], X1_fp[:, 1], '\*', alpha=alpha, color="#000099",markeredgecolor='k')

        nx, ny = 200, 100
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim() #坐标轴上的最小值，最大值，分别有x和y轴
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny)) # 生成测试集数据；linspace表示指定间隔内返回均匀间隔的数字
        Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z = Z[:, 1].reshape(xx.shape # 取一个类的可能概率，重新整形显示

        plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes', norm=colors.Normalize(0., 1.))
        plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='k') # 轮廓线绘制

        plt.plot(lda.means_[0][0], lda.means_[0][1], 'o', color='black', markersize=10, markeredgecolor='k')
        plt.plot(lda.means_[1][0], lda.means_[1][1], 'o', color='black', markersize=10, markeredgecolor='k')

        return splot

    def plot_ellipse(splot, mean, cov, color):
        v, w = linalg.eigh(cov) # linalg是一个函数库，eigh获取特征值，特征向量；返回的内容是(特征向量，特征值)
        u = w[0] / linalg.norm(w[0]) # linalg = linear + algebra表示线性代数，norm涉及到线性代数的概念“范数”
        angle =  np.arctan(u[1] / u[0]) # 反正切
        angle = 180 * angle / np.pi
        ell = mpl.patches.Ellipse(mean, 2 * v[0] ** 0.5, 2 * v[1] ** 0.5, 180 + angle, facecolor=color, edgecolor='yellow', linewidth=2, zorder=2) # 补丁，椭圆形状的补丁
        ell.set_clip_box(splot.bbox) # set_clip_box表示设置艺术剪辑，传入参数是画布的一个类，表示把这个椭圆放到这个bbox里，这里看成是如果显示剪辑就一定要做的一步吧
        ell.set_alpha(0.5)
        splot.add_artist(ell) # 再把这个包装好的椭圆放到画布上
        splot.set_xticks(())
        splot.set_yticks(())

    def plot_lda_cov(lda, splot):
        plot_ellipse(splot, lda.means_[0], lda.covariance_, 'red')
        plot_ellipse(splot, lda.means_[1], lda.covariance_, 'blue')

    def plot_qda_cov(qda, splot):
        plot_ellipse(splot, qda.means_[0], qda.covariances_[0], 'red')
        plot_ellipse(splot, qda.means_[1], qda.covariances_[1], 'blue')

    for i, (X, y) in enumerate([dataset_fixed_cov(), dataset_cov()]):
        lda = LinearDiscriminantAnalysis(solver='svd', store_covariance=True)
        y_pred = lda.fit(X,y).predict(X)
        splot = plot_data(lda, X, y, y_pred, fig_index= 2 * i + 1)
        plot_lda_cov(lda, splot)
        plt.axis('tight') # 具体的表示我们概念里熟知的轴的概念；自动的将坐标轴调整到紧凑型

        qda = QuadraticDiscriminantAnalysis(store_covariance=True)
        y_pred = qda.fit(X,y).predict(X)
        splot = plot_data(qda, X, y, y_pred, fig_index=2 * i + 2)
        plot_qda_cov(qda, splot)
        plt.axis('tight')
    plt.suptitle('Linear Discriminant Analysis vs Quadratic Discriminant''Analysis') # 如此直接使用两个引号相连等价于字符串拼接


### 英语生词

|Quadratic|二次方的|
|Discriminant|判别式|
|covariance|协方差|
|ellipsoid|橄榄球|
|deviation|偏离|
|decision boundary|判定边界|
|eigenvalue|特征值|
|algebra|代数|
|transformation|变换，geometric transformation几何变换|

### 数学名词解释
#### 协方差
+ 衡量两个变量的总体误差
+ 方差是协方差的一种特殊情况，即当两个变量是相同的情况。
+ 这里的两个变量可以不看成是坐标轴里的x和y，而就是两个代号，但数量只限定为2个
+ 协方差公式：cov(X,Y)=E((X-μ)(Y-ν))=E(X•Y)-μν, 中间的点在mac上通过option+8输入
+ 方差公式：var(X) = cov(X,X) = E((X-μ)(X-μ)) = E(X^2)-2E(X)E(X)+(E(X))^2 = E(X^2) - (E(X))^2
+ 涉及到一个数学概念：常数乘以期望可以看成0。
+ 协方差表示变量在两个方向上的误差，这两个方向的变化规律可能一样，也可能不一样
+ 方差就表示两个方向变化规律一样的情况。
+ 经过理解后，原来协方差说的是坐标轴里y的情况，就坐标轴而言，表示的是相同维度下的不同曲线间的关系，如果有多条曲线，则是取任意两条之间的关系。
+ 总结：协方差为正，说明两条曲线X,Y同向变化，协方差越大说明同向程度越高；如果协方差为负，说明X,Y反向运动，协方差越小说明反向程度越高。

#### 相关系数
+ 相关系数可以看成一种协方差
+ 剔除了x,y两个变量量纲影响，标准话之后的特殊协方差。
+ 是协方差，就可以反应两个变量变化是同向还是反向，如果同向变化就为正，反向变化就为负。
+ 由于他是标准化后的协方差，因此更重要的特性来了：它消除了两个变量变化幅度的影响，而只是单纯反映两个变量每单位变化时的相似程度。
+ 新名词：变量的 **变化幅度**
+ 相关系数就是要把变化幅度从协方差的影响中剔除。
+ 问题：为什么能通过处理标准差的方式来剔除变化幅度的影响？
+ 答：标准差描述了变量在整体变化过程中偏离均值的幅度。协方差除以标准差，也就是把协方差中变量变化幅度对协方差的影响剔除掉，这样协方差也就标准化了，它反映的就是两个变量每单位变化时的情况。
+ 相关系数的含义：反映两个变量每单位变化时的情况。
+ 当X或Y的波动幅度变大时，它们的协方差会变大，标准差也会变大，这样相关系数的分子分母都变大，其变大的趋势就会被抵消掉，变小的情况亦然。
+ 所以相关系数的取值范围只能在+1到-1之间。
>总结：
>对于两个变量X,Y
>当他们的 ***相关系数为1*** 时，说明两个变量变化时的正向相似度最大。
> + 即，你变大一倍，我也变大一倍；你变小一倍，我也变小一倍。
> + 是完全 ***正*** 相关
> + 以X,Y为横纵坐标轴，可以画出一条斜率为正数的直线，所以X,Y是线性关系。
>随着他们相关系数减小，两个变量变化时的相似度也变小.
>当 ***相关系数为0*** 时, 两个变量的变化过程没有任何相似度，也即两个变量无关。
>当 ***相关系数为-1*** 时，说明两个变量变化的反向相似度最大。
> + 即，你变大一倍，我变小一倍；你变小一倍，我变大一倍。
> + 是完全 ***负*** 相关
> + 以X,Y为横纵坐标轴，可以画出一条斜率为负数的直线，所以X,Y也是线性关系的。


#### 标准差
+ 标准差描述了变量在整体变化过程中偏离均值的幅度。
+ E()表示相加后再求平均的操作。因此括号中的内容是一个序列，而不仅仅是一个值。E()也可以看成是一个批处理。
+ σx = √[E((X-μ)^2)]

|求平方|在变量值与均值反向偏离时，把负号消除。<br>后面求平均时，每一项数值才不会被正负抵消掉。最后求出的平均值才能更好的体现出每次偏离均值的情况|
|开根号|为了消除负号，把差值进行了平方。那就要把求出的均值开方，将偏离均值的幅度还原回原来的量级|

#### matplotlib里的画布
+ https://pic3.zhimg.com/80/v2-3c0d7b48041864e265f752989385c54a_hd.png
+ figure表示整个画布，是最大的集
+ Axes更接近子图的意思，或者是子画布
+ axis表示在子画布里真正的x轴和y轴

#### 待解决问题
v, w = linalg.eigh(cov)
u = w[0] / linalg.norm(w[0])















end
