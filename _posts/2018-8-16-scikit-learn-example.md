---
layout: post
title:  "scikit learn: Partial Dependence Plots"
date:   2018-08-16 11:17:30
tags: [机器学习, 数据挖掘, scikit-learn, ensemble methods]
---

    from __future__ import print_function
    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt

    from mpl_toolkits.mplot3d import Axes3D

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble.partial_dependence import plot_partial_dependence
    from sklearn.ensemble.partial_dependence import partial_dependence
    from sklearn.datasets.california_housing import fetch_california_housing

    def main():
        cal_housing = fetch_california_housing()

        X_train, X_test, y_train, y_test = train_test_split(cal_housing.data, cal_housing.target, test_size=0.2, random_state=1)

        names = cal_housing.feature_names

        print("Training GBRT...")
        clf = GradientBoostingRegressor(n_estimators=100, max_depth=4, Learning_rate=0.1, loss='huber', random_state=1)
        clf.fit(X_train, y_train)
        print("done.")

        print('Convenience plot with ``partial_dependence_plots``')

        features = [0, 5, 1, 2, (5, 1)]
        fig, axs = plot_partial_dependence(clf, X_train, features, feature_names=names, n_jobs=3, grid_resolution=50)

        fig.suptitle('Partial dependence of house value on nonlocation features\\n for the California housing dataset')

        plt.subplot_adjust(top=0.9)

        print('Custom 3d plot via ``partial_dependence``')
        fig = plt.figure()

        target_feature = (1, 5)
        pdp, axes = partial_dependence(clf, target_feature, X=X_train, grid_resolution=50)
        XX, YY = np.meshgrid(axes[0], axes[1])
        Z = pdp[0].reshape(list(map(np.size, axes))).T
        ax = Axes3D(fig)
        surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=plt.cm.BuPu, edgecolor='k')
        ax.set_xlabel(names[target_feature[0]])
        ax.set_ylabel(names[target_feature[1]])
        ax.set_zlabel('Partial dependence')

        ax.view_init(elev=22, azim=122)
        plt.colorbar(surf)
        plt.suptitle('Partial dependence of house value on median\\n age and average occupancy')
        plt.subplots_adjust(top=0.9)

        plt.show()

    if __name__ == '__main__':
        main()

### 背景介绍
三维图表挺吸引人的。
这个模型图解了模型和重要特征的关系，将重要特征的数据走向以图表的形式体现出来。

end
