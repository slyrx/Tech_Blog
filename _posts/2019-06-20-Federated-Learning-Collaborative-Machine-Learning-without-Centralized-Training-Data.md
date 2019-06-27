---
layout: post
title:  "联合学习: 无需集中训练的协作机器学习"
date:   2019-06-20 10:11:30
tags: [联合学习, tensorflow, 翻译, TFUG]
---

[联合学习: 无需集中训练的协作机器学习](http://ai.googleblog.com/2017/04/federated-learning-collaborative.html "联合学习: 没有集中训练的协作机器学习")
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

***导语:***
本文翻译自 Google AI 团队科学家 Brendan McMahan 和 Daniel Ramage 的博客文章[ Federated Learning: Collaborative Machine Learning without Centralized Training Data ](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)。
  
标准的机器学习方法需要在一台机器上或一个数据中心中有集中的训练数据。为了使谷歌的服务更加完善，我们已经构建了最安全和最具稳定性的云基础架构来处理这些数据。现在，对于通过用户与移动设备交互训练的模型，我们正在引入一种新的方式：_联合学习_。
  
联合学习可以使移动设备协同训练一个共享的预测模型，同时将所有的训练数据保存在设备上，籍此将机器学习和云端存储数据解耦合。这超越了使用本地模型在移动设备上进行预测的方式 (比如 [Mobile Vision API](https://developers.google.com/vision/) 和将模型引入设备 _训练_ 的 [On-Device Smart Reply](https://research.googleblog.com/2017/02/on-device-machine-intelligence.html))。 
  
它的工作过程如下：在设备中下载当前模型，通过在设备上学习数据来改进它，然后将改动总结成一个集中的小的更新。只有这个更新会发送到云端，使用加密技术传输。在云端它会立即与其他用户的更新进行平均处理，以改进共享模型。所有的训练数据还保持在本地设备上，并不会被发送到云端存储。

[![](https://1.bp.blogspot.com/-K65Ed68KGXk/WOa9jaRWC6I/AAAAAAAABsM/gglycD_anuQSp-i67fxER1FOlVTulvV2gCLcB/s640/FederatedLearning_FinalFiles_Flow%2BChart1.png)](https://1.bp.blogspot.com/-K65Ed68KGXk/WOa9jaRWC6I/AAAAAAAABsM/gglycD_anuQSp-i67fxER1FOlVTulvV2gCLcB/s1600/FederatedLearning_FinalFiles_Flow%2BChart1.png)

<center>本地设备会根据使用情况(A)在本地对模型进行个性化设置。许多用户更新的聚合 (B) 形成了对共享模型一致的修改(C)，之后这个过程会被重复。</center>

联合学习可以实现更智能的模型，更低的延迟和更低的功耗，同时还确保隐私。这种方法还有另一个直接的好处：除了给共享模型提供更新，本地设备中的改进模型也可以被立即使用，通过操作手机的方式来增强个性化的体验。
  
我们现在正在 [Gboard on Android ](https://blog.google/products/search/gboard-now-on-android/)针对 Google 键盘测试联合学习。当 Gboard 显示建议的查询时，本地设备会在存储有关当前上下文的信息以及是否点击了该建议。联合学习通过处理设备上的历史记录以建议改进下一个迭代的 Gboard 查询建议模型。

[![](https://1.bp.blogspot.com/-W-husQJfa7s/WObDco6Ql0I/AAAAAAAABso/ERk3Q3mM2xILzEgMa0RMi5UJED7VDLYCACLcB/s640/2017-04-06.gif)](https://1.bp.blogspot.com/-W-husQJfa7s/WObDco6Ql0I/AAAAAAAABso/ERk3Q3mM2xILzEgMa0RMi5UJED7VDLYCACLcB/s1600/2017-04-06.gif)

为了实现联合学习，我们必须要克服许多算法和技术的挑战。在传统的机器学习系统，像[随机梯度下降](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) ( SGD )这样运行大型数据集的优化算法均匀分区在云服务器上。这种高度迭代的算法在训练数据方面需要低延时和高吞吐量。在联合学习的设置中，数据以极不均匀的方式分布在数百万台设备上。除此之外，这些设备具有明显更高的延时，更低的吞吐量，并且只能间歇的用于训练。
  
这些带宽和延迟的限制激发了我们提出 [平均联合算法](https://arxiv.org/abs/1602.05629)，相较于原生的联合 SGD 版本，它可以使用10-100倍的通信速度训练深度网络。处理的关键是使用现代移动设备中强大的处理器去计算比简单梯度处理更重要的高质量更新。鉴于它是通过使用少量迭代高质量更新去产生一个优质模型的，训练需要的通信就会少很多。因为通常上传速度会比下载速度[小很多](http://www.speedtest.net/reports/united-states/)，我们还开发了一种叫[压缩更新](https://arxiv.org/abs/1610.05492)的新方式来降低100倍的通信成本，它的核心是随机旋转和量化。这些方法的关注点在训练深度网络，我们也为高维稀疏凸模型[设计算法](https://arxiv.org/abs/1610.02527)，用于计算像点击率预测这样的问题。

将此技术部署到数百万的运行 Gboard 的设计各异的电话需要积累多年的技术栈。在设备训练方面使用微型版本的[ TensorFlow ](https://www.tensorflow.org/)。
仔细的安排调度，确保训练仅在设备空闲的时候运行，插入电源并连接无限网络，将不会对设备的性能造成影响。

[![](https://3.bp.blogspot.com/-sb40Lg5MchE/WOa92rSrXVI/AAAAAAAABsU/edvH01nn2SMCnJle8mLnHT_hCa-xXrnsACLcB/s640/02_Personalization%2Bsleeping.png)](https://3.bp.blogspot.com/-sb40Lg5MchE/WOa92rSrXVI/AAAAAAAABsU/edvH01nn2SMCnJle8mLnHT_hCa-xXrnsACLcB/s1600/02_Personalization%2Bsleeping.png)

本地设备仅在使用体验不受到负面影响的情况下才参与联合学习。系统之后将以安全、高效、可扩展和容错的方式进行通信和聚合模型的更新，只有将研究和基础设备结合才能让联合学习发挥优势成为可能。

我们并没有止步于联合学习无需在云端存储用户数据。我们使用加密技术开发了一个[安全聚合协议](http://eprint.iacr.org/2017/281)，因此协调服务器只会在100或1000个用户都参与时解密平均更新，没有独立的设备会在平均前被检查。它是同类中第一个适用于深度网络规模问题和实际连接约束的协议。我们设计了平均联合，因此协调服务器只需要平均更新，它允许使用安全的聚合方式。这个协议是通用的，也可以应用于其他问题。我们正在努力实现此协议的产品级实施，并期望在不久的将来将其部署到联合学习的应用程序中。
  
我们的工作只触及了这一可能性的表面。联合学习不能解决所有机器学习的问题（例如，通过使用精心标注的例子训练学习[识别不同狗的品种](https://research.googleblog.com/2016/08/improving-inception-and-image.html)），而其它模型的必要训练数据则存储到了云端(例如为 Gmail 过滤垃圾邮件做训练)。因此 Google 将继续推进云端机器学习的最新技术，我们也会继续研究扩展使用联合学习解决问题的领域。除了 Gboard 查询推荐外，我们希望通过基于设备上实际输入的内容（它可以是任何风格）、喜欢用什么样的方式查看、分享和删除并排序照片来改进语言模型。
  

应用联合学习需要机器学习从业者接受新的工具和新的思考方式，模型开发、训练和无法直接访问原始数据标注的评估，同时通信成本也是一个制约因素。我们相信联合学习的用户利益紧密结合了相关的技术挑战，通过发布我们的工作内容希望在机器学习社区内展开广泛的讨论对话。
  
**致谢** <br>
这篇文章反映了谷歌研究中许多人的工作，包括 BlaiseAgüerayArcas，Galen Andrew，Dave Bacon，Keith Bonawitz，Chris Brumme，Arlie Davis，Jac de Haan，Hubert Eichner，Wolfgang Grieskamp，Wei Huang，Vladimir Ivanov，Chloé Kiddon，JakubKonečný，Nicholas Kong，Ben Kreuter，Alison Lentz，Stefano Mazzocchi，Sarvar Patel，Martin Pelikan，Aaron Segal，Karn Seth，Ananda Theertha Suresh，Iulia Turc，Felix Yu，Antonio Marcedone 以及我们在 Gboard 团队的合作伙伴。
