---
layout: post
title:  "scikit learn: Kernel Density Estimation"
date:   2018-07-26 15:19:30
tags: [机器学习, 数据挖掘, scikit-learn, Nearest Neighbors]
---

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.datasets import load_digits
    from sklearn.neighbors import KernelDensity
    from sklearn.decomposition import PCA
    from sklearn.model_selection import GridSearchCV

    digits = load_digits()
    data = digits.data

    pca = PCA(n_components=15, whiten=False) # 主成分分析
    data = pca.fit_transform(digits.data)

    params = {'bandwidth': np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params) # 核密度评估
    grid.fit(data)

    print("best bandwidth:{0}".format(grid.best_estimator_.bandwidth))

    kde = grid.best_estimator_ # 使用最优的估计量计算核密度估计

    new_data = kde.sample(44, random_state=0) # 从样例中取出44个点
    new_data = pca.inverse_transform(new_data)

    new_data = new_data.reshape((4, 11, -1)) # 将数据整理成4x11的网格
    real_data = digits.data[:44].reshape((4, 11, -1))

    fig, ax = plt.subplots(9, 11, subplot_kw=dict(xticks=[], yticks=[])) # 画出实际的数字和重新采样的数字，ax为小子图画布

    for j in range(11):
        ax[4, j].set_visible(False) # 设置第4+1行的子图为不可见
        for i in range(4):
            im = ax[i, j].imshow(real_data[i, j].reshape((8, 8)),cmap=plt.cm.binary, interpolation='nearest')
            im.set_clim(0, 16)
            im = ax[i + 5, j].imshow(new_data[i, j].reshape((8, 8)),cmap=plt.cm.binary, interpolation='nearest')
            im.set_clim(0, 16)

    ax[0, 5].set_title('Selection from the input data')
    ax[5, 5].set_title('"New" digits drawn from the kernel density model')

    plt.show()

### 背景介绍
这个示例是为了在样本缺少的情况下，构造新的符合条件的样本而创建的。本质上还是一个创造数据集的函数。

#### 名词解释
PCA Principal components analysis 主成分分析，一种分析，简化数据集的技术

end
