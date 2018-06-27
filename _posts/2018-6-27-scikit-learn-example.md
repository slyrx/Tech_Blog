---
layout: post
title:  "scikit learn: Plot classification probability"
date:   2018-06-27 14:01:30
tags: [机器学习, 数据挖掘, scikit-learn]
---

    print(__doc__)

    import matplotlib.pyplot as plt
    import numpy as np

    from sklearn.linear_model import logisticRegression
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn import datasets

    iris = datasets.load_iris() # 数据使用鸢尾花数据集。包含150中鸢尾花的信息，没50中取自三个鸢尾花种之一。
    X = iris.data[:, 0:2] # 只取数据的两个特征做本实验
    y = iris.target # 赋值类别标签

    n_features = X.shape[1] # 提取特征个数
    C = 1.0
    kernel = 1.0 * RBF([1.0, 1.0]) # Radial basis function kernel 径向基函数核

    classifier = {'L1 logistic': LogisticRegression(C=C, penalty='l1'),
                  'L2 logistic (OvR)': LogisticRegression(C=C, penalty='l2'),
                  'Linear SVC': SVC(kernel='linear', C=C, probability=True, random_state=0),
                  'L2 logistic (Multinomial)': LogisticRegression(C=C, solver='lbfgs', multi_class='multinomial'), 'GPC': GaussianProcessClassifier(kernel)
                } # 创建不同的分类器，以字典形式存储

    n_classifiers = len(classifiers)

    plt.figure(figsize=(3 \* 2, n_classifiers \* 2)) # 设定画布大小， 6行8列
    plt.subplots_adjest(bottom=.2, top=.95) # .2表示0.2,同理.95表示0.95

    xx = np.linspace(3,9,100) # 从3到9，以等间距的方式分成100份
    yy = np.linspace(1,5,100).T #从1到5，以等间距的方式分成100份，再进行转置
    xx,yy = np.meshgrid(xx, yy) # meshgrid将xx向量变成矩阵，由一维的变成多维的
    Xfull = np.c_[xx.ravel(), yy.ravel()] # ravel将多维数组降为一维，c_串联函数

    for index, (name, classifier) in enumerate(classifiers.items()): # 依次取不同的分类器进行实验
        classifier.fit(X,y)

        y_pred = classifier.predict(X)
        classif_rate = np.mean(y_pred.ravel() == y.ravel()) * 100 # 计算预测值的均值, y_pred.ravel() == y.ravel()结论虽然是True和False，但是通过mean函数还是可以计算出均值，可能是所有True看成是1，False看成是0，相加的总和除以总元素数，得到的结果吧
        print("classif_rate for %s : %f" % (name, classif_rate))

        probas = classifier.predict_proba(Xfull) # 返回预测是某值的概率，一行的和为1，每一列是一个类
        n_classes = np.unique(y_pred).size # 预测值去重处理，去重后的数量
        for k in range(n_classes):
          plt.subplot(n_classifiers, n_classes, index * n_classes + k + 1) #画布分出的区域图，选择第几个显示
          plt.title('Class %d' % k) # 对应的标题
          if k == 0:
              plt.ylabel(name) # y轴的名称
          imshow_handle = plt.imshow(probas[:, k].reshape((100, 100)),  extent=(3,9,1,5), origin='lower') # 把预测概率点的值以图片的形式表现出来，extent表示坐标左下角和右上角的位置,即横轴是从初始开始算起3到9；竖轴从初始开始算起从1到5，data_coordinates表示坐标。origin原点坐标位置，lower表示把原点放到左下角的位置，upper表示左上角的位置。

          plt.xticks(()) # x的刻度显示,设为空()
          plt.yticks(()) # y的刻度显示,设为空()
          idx = (y_pred == k) # 矩阵和单个整型相等，具体比法为，数组里每一个元素都和k值进行一次比较，看是否相等，将bool结果返回到对y_pred对应的数组，里面是True和False
          if idx.any(): # 查看两个矩阵是否由一个对应元素相等
              plt.scatter(X[idx, 0], X[idx, 1], marker='o', c='k') # 散点图，x,y值，marker圆形表示，c颜色选择黑色k; X[idx, 0]表示idx里为真的那些值的下标的集合所有对应的两列中的列1，同理X[idx, 1]，为列2

    ax = plt.axes([0.15, 0.04, 0.7, 0.05]) # 输入为一个矩形，这个矩形的基本信息是[left,bottom,width,height],参照对象是整个画布，每个部分的值都归一化到0到1之间，也就是把整个画布看成是1. 其作用就是规定轴图像的大小
    plt.title("Probability") # 设置标题
    plt.colorbar(imshow_handle, cax=ax, orientation="horizontal") # 彩色条绘制，cax表示被绘制到到位置，orientation表示绘制到方向。

    plt.show()




### 概念介绍
径向基函数核？

#### ***np.meshgrid的作用？***
+ 一维的x = [-2, -1, 0, 1], y = [0, 1, 2]
+ z,s = np.meshgrid(x, y) # 表示把1行4列的x，和1行3列的y，变成，3行重复的x，4列重复的y
+ z = [-2, -1, 0, 1]
+     [-2, -1, 0, 1]
+     [-2, -1, 0, 1]
+ s = [0, 0, 0, 0]
+     [1, 1, 1, 1]
+     [2, 2, 2, 2]
+ z和s都变成了3行4列的矩阵了，只是填充的内容，分别是由x，y各自填充的。

#### ***np.ravel的作用？***
+ x是2x2矩阵，x = [1, 2]
+                [3, 4]
+ x.ravel() # 将多维数组降为一维, 数组中如果由相同值，不会去重
+ x = [1, 2, 3, 4]

#### ***np.c_的作用？***
+ np.c_[[1,2,3], 0, 0, [4,5,6]] # 串联所有信息，为一维的内容
+ [1,2,3,0,0,4,5,6]

#### ***np.mean的作用？***
+ a = [[1, 2], [3, 4]]
+ np.mean(a) , 2x2矩阵的所有元素求和除以个数得到的均值
+ np.mean(a, axis=0) , axis=0表示纵轴列方向，axis=1表示横轴行方向，而mean方法则是单独计算一个方向上的均值，此处为计算竖轴方向的均值
+ np.mean(a, axis=0) ==> [2., 3.] ，得到的均值是浮点精度的。
+ np.mean(a, axis=1) ==> [1.5,3.5]







end
