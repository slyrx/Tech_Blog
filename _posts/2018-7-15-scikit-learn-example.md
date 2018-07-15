---
layout: post
title:  "scikit learn: Compare Stochastic learning strategies for MLPClassifier"
date:   2018-07-15 07:14:30
tags: [机器学习, 数据挖掘, scikit-learn, Neural Networks]
---

print(__doc__)
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets

params = [{'solver':'sgd', 'learning_rate':'constant', 'momentum':0,  'learning_rate_init':0.2},
          {'solver':'sgd', 'learning_rate':'constant', 'momentum':.9, 'nesterovs_momentum':False, 'learning_rate_init':0.2},
          {'solver':'sgd', 'learning_rate':'constant', 'momentum':.9, 'nesterovs_momentum':True,'learning_rate_init':0.2},
          {'solver':'sgd', 'learning_rate':'invscaling','momentum':0, 'learning_rate_init':0.2},
          {'solver':'sgd', 'learning_rate':'invscaling','momentum':0, 'learning_rate_init':0.2},
          {'solver':'sgd', 'learning_rate':'invscaling','momentum':.9,'nesterovs_momentum':False,'learning_rate_init':0.2},
          {'solver':'adam','learning_rate_init':0.01}]

labels = ["constant learning-rate", "constant with momentum", "constant with Nesterov's momentum", "inv-scaling learning-rate", "inv-scaling with momentum", "inv-scaling with Nesterov's momentum", "adam"]

plot_args = [{'c': 'red', 'linestyle':'-'},
            {'c': 'green', 'linestyle':'-'},
            {'c': 'blue', 'linestyle':'-'},
            {'c': 'red', 'linestyle':'--'},
            {'c': 'green', 'linestyle':'--'},
            {'c': 'blue', 'linestyle':'--'},
            {'c': 'black', 'linestyle':'-'},]

def plot_on_dataset(X, y, ax, name):
    print("\nlearning on dataset %s"% name)
    ax.set_title(name)
    X = MinMaxScaler().fit_transform(X) # 通过缩放特征来把每个特征变换到一个给定区间。默认是把序列里最大和最小值到范围设定为给定区间。
    mlps = []
    if name == "digits":
       max_iter = 15
    else:
       max_iter = 400

    for label, param in zip(lables, params): # zip将对象中对应等元素打包成一个个元组，然后返回由这些元组组成的列表。
        print("training: %s" % label)
        mlp = MLPClassifier(verbose=0, random_state=0, max_iter=max_iter, \*\*param) #
        mlp.fit(X, y) #
        mlps.append(mlp) # 这里是追加到一个模型列表中
        print("Training set score : %f" % mlp.score(X, y))
        print("Training set loss: %f" % mlp.loss_)
    for mlp, label, args in zip(mlps, labels, plot_args):
        ax.plot(mlp.loss_curve_, label=label, \*\*args)

fig, axes = plt.subplots(2, 2, figsize=(15, 10)) # axes直接理解成画布，但是它可大可小

iris = datasets.load_iris()
digits = datasets.load_digits()
data_sets = [(iris.data, iris.target), (digits.data, digits.target), datasets.make_circles(noise=0.2, factor=0.5, random_state=1), datasets.make_moons(noise=0.3, random_state=0)]

for ax, data, name in zip(axes.ravel(), data_sets, ['iris', 'digits', 'circles', 'moons']):
    plot_on_dataset(\*data, ax=ax, name=name) # \*data直接包括了X,y

fig.legend(ax.get_lines(), labels, ncol=3, loc="upper center") # 图例函数，占地面积，大小
plt.show()



### 背景介绍
这个例子图示了不同随机学习策略训练出的损失函数曲线，包括SGD和Adam。由于运行时间的关系，这里采用了小数据集L-BFGS。所得的结论可以推演到大数据集上。
需要注意的是，这些结果高度依赖学习率的初始设定值。

#### 术语理解
+ inv-scaling 逆缩放，是learning_rate学习率地一个选项，表示每一步逐渐地降低学习率
+ 有效学习公式: effective_learning_rate = learning_rate_init/pow(t, power_t)
+ 在第t步，使用对应地指数power_t，
+ pow(t, power_t) 返回基数地指数次幂, t的power_t次方，相当于x^y次方

#### 模型参数设置理解
+ 学习率设置为常量constant，动量检验才需要用到nesterovs_momentum是否开启，也就是调换一下求向量的步骤，先距离还是先向量，三角形的另外两个边。如果把学习率设置为invscaling，那么就可以不考虑nesterovs_momentum是否开启这个问题了，因为涉及不到了。

#### 遗留问题
+ 什么是学习率？有效学习率？
+ 参考 http://cs231n.github.io/neural-networks-3/
+ at each time step ‘t’ using an inverse scaling exponent of ‘power_t’.其中的t是什么意思？
+ 结论图标数据解读？

#### 名词解释
SGD: Stochastic Gradient Descent，是一种Gradient Descent(GD)的改良，在GD里输入全部的training dataset,根据累积的loss才更新一次权重，因此收敛速度很慢，SGD随机抽一笔training sample,依照其loss更新权重。
Adam: Adaptive Moment Estimation,是一种自己更新学习速率的方法，会根据GD计算出来的值调整每个参数的学习率(因材施教)。
inv-scaling with momentum: momentum是为了以防GD类的方法陷入局部最小值而衍生的方法，可以利用momentum降低陷入局部最小值的机率，此方法是参考物理学动量的观点。参考：https://www.bookstack.cn/read/machine-learning-python/Neural_Networks-plot_mlp_training_curves.md；http://cs231n.github.io/neural-networks-3/
+ 以上这些 ***改进梯度下降*** 的最佳化方法，都需要设定learning_rate_init值。


#### 英语生词
momentum 动量
invertor 逆转
inverse 倒置
inv 反，倒置
scaling 缩放
gradually 逐渐地
exponent 指数



end
