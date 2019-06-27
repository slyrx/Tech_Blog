---
layout: post
title:  "scikit learn: Kernel Density Estimation of Species Distributions"
date:   2018-07-27 15:00:30
tags: [机器学习, 数据挖掘, scikit-learn, Nearest Neighbors]
---

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_species_distributions
    from sklearn.datasets.species_distributions import construct_grids
    from sklearn.neighbors import KernelDensity

    try:
        from mpl_toolkits.basemap import Basemap
        basemap = True
    except ImportError:
        basemap = False

    data = fetch_species_distributions()
    species_names = ['Bradypus Variegatus', 'Microryzomys Minutus']

    Xtrain = np.vstack([data['train']['dd lat'], data['train']['dd long']]).T

    ytrain = np.array([d.decode(ascii).startswith('micro') for d in data['train']['species']], dtype='int')

    Xtrain \*= np.pi / 180.

    xgrid, ygrid = construct_grids(data)
    X, Y = np.meshgrid(xgrid[::5], ygrid[::5][::-1])
    land_reference = data.coverages[6][::5, ::5]
    land_mask = (land_reference > -9999).ravel()

    xy = np.vstack([Y.ravel(), X.ravel()]).T
    xy = xy[land_mask]
    xy \*= np.pi / 180.

    fig = plt.figure()
    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.05)

    for i in range(2):
        plt.subplot(1, 2, i + 1)

        print(" - computing KDE in spherical coordinates")
        kde = KernelDensity(bandwidth=0.04, metric='haversine', kernel='gaussian', algorithm='ball_tree')

        kde.fit(Xtrain[ytrain == i]) # 抽取数据特征，以便于后续生成的数据也能具有该特征

        Z = -9999 + np.zeros(land_mask.shape[0]) # -9999表示图例里的海洋
        Z[land_mask] = np.exp(kde.score_samples(xy))
        Z = Z.reshape(X.shape)

        levels = np.linspace(0, Z.max(), 25)
        plt.contourf(X, Y, Z, levels=levels, cmap=plt.cm.Reds)

        if basemap:
            print(" - plot coastlines using basemap")
            m = Basemap(projection='cyl', llcrnrlat=Y.min(), urcrnrlat=Y.max(), llcrnrlon=X.min(), urcrnrlon=X.max(), resolution='c')
            m.drawcoastlines()
            m.drawcountries()
        else:
            print(" - plot coastlines from coverage")
            plt.contour(X, Y, land_reference, levels=[-9999], colors='k', linestyles="solid")
            plt.xticks([])
            plt.yticks([])

        plt.title(species_names[i])

    plt.show()

### 背景介绍
示例主要讲了抽取数据特征的kde模型，因为是抽取数据特征的部分，暂时研究的不那么深入。

#### 英语生词
Species 物种

end
