---
layout: post
title:  "scikit learn: Decision boundary of label propagation versus SVM on the Iris datasets"
date:   2018-07-30 20:47:30
tags: [机器学习, 数据挖掘, scikit-learn, Semi Supervised Classification]
---


### 背景解释
在这个示例中，Label propagation是一种半监督的算法，翻译过来被称作标签传播，就是只知道一小部分标签，但是通过这些标签对部分测试做预测，可以计算出这部分测试集的label，之后将新测试集中计算出的label也作为训练集加入到原训练集中，扩大训练集的样本数量。 而与它对比的使用svm算法。

结论是，还是svm效果更好一些，当只有1/3的标注数据时，也同样能对剩余的2/3做适当的区分，并继续进行后面的分类算法。后面的1/2和100%数据，逻辑类似。

end
