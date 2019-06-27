---
layout: post
title:  "scikit learn: Hashing feature transformation using Totally Random Trees"
date:   2018-08-13 18:06:30
tags: [机器学习, 数据挖掘, scikit-learn, ensemble methods]
---

        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn.datasets import make_circles
        from sklearn.ensemble import RandomTreesEmbedding, ExtraTreesClassifier
        from sklearn.decomposition import TruncatedSVD
        from sklearn.naive_bayes import BernoulliNB

        X, y = make_circles(factor=0.5, random_state=0, noise=0.05)

        hasher = RandomTreesEmbedding(n_estimators=10, random_state=0, max_depth=3) # 一群完全随机的树木。无监督地将数据集转换为高维稀疏表示。 
        X_transformed = hasher.fit_transform(X)

        svd = TruncatedSVD(n_components=2) # 这里表示降维后，特征会不再明显了
        X_reduced = svd.fit_transform(X_transformed) # 使用svd进行数据降维，RandomTreesEmbedding表示升维，TruncatedSVD表示降维；singular value decomposition (SVD); 后面的降维数据暂时没有用到，只是将降维后的样子做了简单的展示

        nb = BernoulliNB() # 这句之前的内容都是在对数据进行升维、降维的处理
        nb.fit(X_transformed, y)

        trees = ExtraTreesClassifier(max_depth=3, n_estimators=10, random_state=0)
        trees.fit(X, y)

        fig = plt.figure(figsize=(9, 8))

        ax = plt.subplot(221)
        ax.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor='k')
        ax.set_title("Original Data (2d)")
        ax.set_xticks(())
        ax.set_yticks(())

        ax = plt.subplot(222)
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, s=50, edgecolor='k')
        ax.set_title("Truncated SVD reduction (2d) of transformed data (%dd)" % X_transformed.shape[1])
        ax.set_xticks(())
        ax.set_yticks(())

        h = .01
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        transformed_grid = hasher.transform(np.c_[xx.ravel(), yy.ravel()])
        y_grid_pred = nb.predict_proba(transformed_grid)[:, 1]

        ax = plt.subplot(223)
        ax.set_title("Naive Bayes on Transformed data")
        ax.pcolormesh(xx, yy, y_grid_pred.reshape(xx.shape))
        ax.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor='k')
        ax.set_ylim(-1.4, 1.4)
        ax.set_xlim(-1.4, 1.4)
        ax.set_xticks(())
        ax.set_yticks(())

        y_grid_pred = trees.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        ax = plt.subplot(224)
        ax.set_title("ExtraTrees predictions")
        ax.pcolormesh(xx, yy, y_grid_pred.reshape(xx.shape))
        ax.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor='k')
        ax.set_ylim(-1.4, 1.4)
        ax.set_xlim(-1.4, 1.4)
        ax.set_xticks(())
        ax.set_yticks(())

        plt.tight_layout()
        plt.show()

### 背景介绍
把低维的数据处理成高维的数据。再对高维数据进行分类处理。

#### 结论
将数据从低维提到高维后，经过分类器再进行分类，结果更加鲜明。

end
