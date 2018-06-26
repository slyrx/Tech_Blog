---
layout: post
title:  "scikit learn: Recognizing hand-written digits"
date:   2018-06-26 07:40:30
tags: [机器学习, 数据挖掘, scikit-learn]
---

    print(\__doc__) #输出文件开头的注释内容

    import matplotlib.pyplot as plt # 绘图库

    from sklearn import datasets, svm, metrics # 数据集，支持向量机，矩阵等库

    digits = datasets.load_digits() # 加载数据集
    images_and_labels = list(zip(digits.images, digits.target)) # zip将对象中对应等元素打包成一个个元组，然后返回由这些元组组成等列表。"tuple"称为元组，和list类似，其中内容不得修改，在里面可以放入字典。

    for index, (image, label) in enumerate(images_and_labels[:4]):# 这里为了简略，只取了4个图片的数据量
    \# index不是关键词，表示直接提取了images_and_labels里的下标序号，主要是因为enumerate里返回的下标，所以它能表示。
    \# enumerate 枚举，用于将一个可遍历的数据对象组合为一个索引序列，同时列出数据和数据下标，一般用在for循环当中。
    \# feature表示一个大画布，plt表示选择好的一个小的区域，plt指的标题、轴和标签这些都是指这个小区域里的内容。大画布里可以有多个标题、轴和标签。
        plt.subplot(2, 4, index + 1) # 绘制小图，将图像画布分为2行4列，当前位置为1，起始值为1.
        plt.axis('off') # 关闭轴线和标签显示，如果只有轴线和标签，没有其他内容，就会什么都不显示
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest') # 利用matplotlib包对图片进行绘制，image要绘制对图像或数组，cmap颜色图谱，interpolation表示图片以什么方式展现，像素或高斯模糊等15种方式。
        plt.title('Training: %i' % label) #子图像标题

    n_samples = len(digits.images) # 设置图片个数，也就是样本个数
    data = digits.images.reshape((n_samples, -1)) # reshape 列数处显示-1，理解其含义为把当前矩阵由3列压缩成2列，就是在原来的基础上减少1列。如果是行数，同理。

    classifier = svm.SVC(gamma=0.001) #定义离散型支持向量机，gamma隐含的决定了数据映射到新到特征空间后到分布，gamma越大，支持向量越少，gamma越小,支持向量越多。支持向量到个数影响训练和预测速度。还有一个参数C，表示惩罚系数，即对误差对宽容度，C越高，说明越不能容忍出现误差，容易过拟合；C越小，容易欠拟合。C过大或过小，泛化能力变差。

    classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2]) # 数组data[:n]表示了data变量里的所有行; //符号表示取整除-返回商的整数部分，这里表示选择样本的一半来做这个实验

    expected = digits.target[n_samples // 2:] # 相当于抽取的测试集的真实值
    predicted = classifier.predict(data[n_samples // 2:]) # 对分割的测试集进行预测

    print('Classification report for classifier %s:\n%s\n' % (classifier, metrics.classification_report(expected, predicted)))# 打印出分类器信息，metrics表示度量，是sklearn里专门用于打分的类，classification_report用于生成分类器报告，通过真实值和预测值的统计

    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted)) #confusion_matrix表示混淆矩阵，又称为可能性表格或是错误矩阵，是一种特定的矩阵用来呈现算法性能的可视化效果。其每一列代表 **预测值**，每一行代表的是实际 **类别**

    images_and_predictions = list(zip(digits.images[n_samples //2:], predicted))
    for index, (image, prediction) in enumerate(images_and_predictions[:4]):
        plt.subplot(2, 4, index + 5)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Prediction: %i' % prediction) #打印变量内容时，要在显示字符串和变量之间加上一个%百分号。

    plt.show()




### 识别手写体
#### __需要确定out的输出是什么？__
+ 输出了分类器的参数信息
+ 输出了分类器的报告，预测值和实际值的对比统计打分情况
+ 输出了由预测值和真实值生成的混淆矩阵


#### __输入数据是什么？__
图片的矩阵，一个图片有8行8列，多个这样的图片，对应对label是数字
python语法：
+ [:-1] Python syntax, which produces a new array that contains all but the last entry of digits.data

#### __什么是混淆矩阵？__
混淆矩阵是分析分类器识别不同类元组的一种游泳的工具。
__TP__ 和 __TN__ 告诉我们分类器何时分类正确，而 __FP__ 和 __FN__ 告诉我们分类器何时分类错误。

|      | yes | no | 合计 |
| yes  | TP | FN | P |
| no   | FP | TN | N |
| 合计 | P' | N' | P + N |

给定m个类(其中m>=2), 混淆矩阵(confusion matrix)是一个至少为m x m的表。
前m行和m列只能够的表目CM(i,j),指出类i的元组被分类器标记为类j的个数。
理想地，对于具有 ___高准确率的分类器___ ，__大部分元组应该被混淆矩阵从CM(1,1)到CM(m,m)的对角线上的表目表示，而其他表目为0或者接近0.__ 也就是说，FP和FN接近0.

该表可能有附加的行和列，提供合计。例如，在图8.14的混淆矩阵中，显示了P和N。此外，P'是被分类器标记为正的元组数(TP + FP), N'是被标记为负的元组数(TN + FN).
元组的总数为 TP + TN + FP + PN , 或 P + N, 或 P' + N'. 注意，尽管所显示的混淆矩阵是针对二元分类问题的，但是容易用类似的方法给出多分类问题的混淆矩阵。

通过混淆矩阵可以统计得出 ***准确率***。























end
