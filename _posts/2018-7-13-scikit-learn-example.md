---
layout: post
title:  "scikit learn: Varying regularization in Multi-layer Perceptron"
date:   2018-07-13 09:36:30
tags: [机器学习, 数据挖掘, scikit-learn, Neural Networks]
---

print(__doc__)

import numpy as np
from matplotlib import pyplot as plt
from matplotllib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier

h = .02
alphas = np.logspace(-5, 3, 5) # logspace创建等比函数，默认是以10为底数，-5和3都表示幂，以-5为幂的起始，即10的-5次方，0.00001，以3为幂结束，即10的3次方，1000，-5和3之间按照5等分生成5个数，分别是10的5种次方，组成一个数组。另外还可以通过base=参数指定底数，比如将底数由10改成2，base=2. 对应的linspace是等差函数
names = []
for i in alphas:
    names.append('alpha' + str(i))

classifier = []
for i in alphas:
    classifier.append(MLPClassifier(alpha=i, random_state=1))

X,y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=0, n_clusters_per_class=1) # 生成数据

rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape) # rng.uniform返回指定范围内的随机数，这里将随机数分别乘到X(100，2)的矩阵元素上，结果在放大2倍，追加到X上面
linearly_separable = (X, y) # 将X，y组合成一个tuple，标注对应属性值

datasets = [make_moons(noise=0.3, random_state=0), make_circles(noise=0.2, factor=0.5, random_state=1), linearly_separable] # 生成一个list，将moon、circles和线性数据混合起来了。

figure = plt.figure(figsize=(17.9))
i = 1

for X, y in datasets:
    X = StandardScaler().fit_transform(X) # StandardScaler去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本。fit_transform表示先拟合数据，然后将其转化为标准形式。transform表示通过找中心和缩放等实现标准化。
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) # 以乘倍的方式生成新的数据， 这里np.arange里的参数h表示step的意思

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifier) + 1, i)
    ax.scatter(X_train[:, 0], X_train[:,1], c=y_train, cmap=cm_bright)
    ax.scatter(X_test[:,0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    for name, clf in zip(names, classifier):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        if hasattr(clf, "decision_function"): # 什么情况下有decision_function? decision_function是找到超平面的意思。只有svm支持向量机分类算法才用到
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1] #其他分类算法一般是输出预测概率，即predict_proba。这里clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])输出的是一个<type 'tuple'>: (67456, 2)，是两列，两列上每行的元素相加之和都为1.老是忘了最初分的是哪两个类？应该是0和1两个类，这里是取的被判断为1的概率，我可以说因为是二分类问题，所以取列为0也是可以的。因为结果就是对称的。而这里参数传入yy.ravel()是把xx和yy拼接起来都作为特征输入了，结果得出了根据这两个拼接后都矩阵判断出都列都归属概率。这里都xx和yy的含义是，横坐标和竖坐标的意思，yy不表示已经预测出的结果，所以并没有把测试的结果提前告知。

        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8) # contourf绘制等高线。

        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='black', s=25) # scatter里的s参数，表示画出来的单个散点的大小，默认是20。当输入是一个数组时，利用数组矩阵的映射机制，则可以同时将x中的点以不同的大小同时表示出来。相同的用法可以推广到c，即color的显示。参考https://blog.csdn.net/u013634684/article/details/49646311
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='black', s=25)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right') # az.text中x,y表示坐标的位置，以0到1内到百分比来确定大概的位置。一篇讲的比较详细的text相对位置的博文 https://blog.csdn.net/qq_31192383/article/details/54380736
        i += 1

figure.subplots_adjust(left=.02, right=.98) # 子图调节函数，left表示子图(subplot)距画板(figure)左边的距离。right表示距右边的距离，同理bottom是底部，top是顶部。wspace表示子图水平间距，hspace表示子图垂直间距。
plt.show()



### 背景介绍
该实验是合成数据集上使用不同正则化参数‘alpha’的比较。图片显示了不同alpha值对应的不同决策方法，每个方法对应的是不同的函数。

alpha是正则化项参数，也就是惩罚项，它通过约束权重的大小来对抗过度拟合。增加alpha可以通过鼓励更小的权重来修复 ***高方差(过度拟合的迹象)*** ，从而导致出现具有较小曲率的决策边界图。类似地，降低alpha可以通过鼓励更大的权重来修复 ***高偏差(欠拟合的迹象)***，可能导致更复杂的决策边界。

#### 问题
+ 为什么正则化参数也叫惩罚系数？
+ 机器学习中几乎都可以看到损失函数后面会添加一个额外项，常用都额外项一般有两种，英文称为l1-norm和l2-norm，中文称作L1正则化和L2正则化，或者L1范数和L2范数。所谓“惩罚”是指对损失函数中对某些参数做一些限制。
+ 对于线性回归模型，使用L1正则化的模型叫做lasso回归；使用L2正则化的模型叫做Ridge回归，也叫“岭回归”。
+ 所以惩罚系数是针对损失函数来谈的，损失函数又是针对正则化来谈的。
+ 惩罚系数也只在回归系列的算法中涉及。
+ L1就是α||ω||1; L2就是α||ω||2^2
+ StandardScaler里的fit_transform和transform的区别是什么？
+ 为什么在标准数据的时候不使用fit_transform？
+ https://blog.csdn.net/quiet_girl/article/details/72517053
+ https://i.stack.imgur.com/PiaIX.png
+ 同一个模型，通过训练集已经确定好了均值μ和方差σ^2，因此在测试集就不需要在fit了，这个fit就是寻找均值μ和方差σ^2的过程。
+ 数据归一化的公式：x'=(x-μ)/σ^2, 每个单独的元素和期望之间的差别，通过除的方法把序列的平均值方差σ^2除掉，也就把序列中一致的部分去掉了，只留下了不一致的部分。虽然除的是方差，但因为方差就是标准差的平方。所以效果与处理标准差相当。都是处理统一的部分。最终结果会落在(0,1)之间。
+ predict、decision_function和predict_proba的区别？
+ predict 输出是预测值，输入是正常特征值
+ predict_proba 输出是属于所有类的概率，是个矩阵，输入是正常特征值
+ decision_function 专用与svm表示超平面的函数
+ 正则化和归一化的区别？
+ 标准化是对数据进行按比例缩放，使之落在一个小对特定区间
+ 正则化：用一组与原问题相“邻近”的解，去逼近原问题的解。
+ 归一化是为了消除不同数据之间的量纲，方便数据比较和共同处理。
+ 标准化是为了方便数据的下一步处理，而进行的数据缩放等变换，并不是为了方便与其他数据一同处理或比较。
+ 正则化是利用先验知识，在处理过程中引入正则化因子，增加引导约束的作用。
+ 理解正则化就是约束的意思。

#### 名词解释
+ 曲率 曲率表示曲线偏离直线的程度。曲率越小，越接近直线；曲率越大，越不接近直线。https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Osculating_circle.svg/600px-Osculating_circle.svg.png
+ 高方差 表示所有样本点比较散，各个值和平均值之间的差距比较大，对应的 **低方差** 就是，比较集中，各个值和平均值之间差距比较小。此时过度拟合是说，因为样本点本身比较散，适应了这种情况，当遇到另一个比较散的样本，就会无法适应。而低方差的样本点比较集中，计算得出的模型，即使遇到另一个比较集中的样本群，也能很好的把它捕捉到，所以这时它的泛化能力就可以被认为很好。
+ 高偏差 字面意思，结论优先选取低方差、低偏差的样本群，高偏差的样本点最不受欢迎，即使是低方差的。
+ https://upload-images.jianshu.io/upload_images/3985559-017412076e8e624e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/475
+ https://upload-images.jianshu.io/upload_images/3985559-b97492337331b562.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/369
+ 正则化参数
+ https://upload-images.jianshu.io/upload_images/3985559-876b152ea262133c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/700
+ https://upload-images.jianshu.io/upload_images/3985559-be2bf612872af0dc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/313

#### 等高线绘制专题
https://morvanzhou.github.io/tutorials/data-manipulation/plt/3-3-contours/

#### 英语单词
+ Perceptron 感知机
+ synthetic 合成的，假的
+ curvature 弯曲程度，曲率
+ aka 是also known as的缩写
+ constrain 约束，压抑
+ bias 偏向，倾向





end
