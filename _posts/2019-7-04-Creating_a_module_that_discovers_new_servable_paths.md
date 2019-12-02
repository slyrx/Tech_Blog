---
layout: post
title:  "Creating a module that discovers new servable paths"
date:   2019-07-04 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第八篇。
        本文档说明了如何扩展 TensorFlow Serving 以监视不同的存储系统，以发现要服务的新（版本）模型或数据。

本文档说明了如何扩展TensorFlow Serving以监视不同的存储系统，以发现要服务的新（版本）模型或数据。特别是，它涵盖了如何创建和使用一个模块来监视存储系统路径以查看新子路径的出现，其中每个子路径代表要加载的新可服务版本。这种模块称为Source <StoragePath>，因为它发出类型为StoragePath（类型定义为字符串）的对象。它可以与SourceAdapter组成，该SourceAdapter从源发现的给定路径创建可服务的Loader。

首先，关于一般性的说明
不需要使用路径作为可处理数据的句柄；它仅说明了一种将可食用物品摄入系统的方法。即使您的环境没有在路径中封装可服务的数据，本文档也会使您熟悉关键的抽象。您可以选择为适合您的环境的类型（例如RPC或pub / sub消息，数据库记录）创建Source <T>和SourceAdapter <T1，T2>模块，或者仅创建一个整体Source <std :: unique_ptr <装载机>>，可直接发出可用的装载机。

当然，无论您的源发出什么类型的数据（无论是POSIX路径，Google Cloud Storage路径还是RPC句柄），都需要有随附的模块，这些模块能够基于该模块加载可服务对象。这样的模块称为SourceAdapters。 Custom Servable文档中描述了如何创建自定义文件。 TensorFlow Serving随附一项服务，用于根据TensorFlow支持的文件系统中的路径实例化TensorFlow会话。可以通过扩展RandomAccessFile抽象（tensorflow / core / public / env.h）向TensorFlow添加对其他文件系统的支持。

本文档重点介绍在TensorFlow支持的文件系统中创建发出路径的源。最后以如何使用您的源代码以及预先存在的模块来为TensorFlow模型提供服务的演练结尾。

创建您的源
我们有一个Source <StoragePath>的参考实现，称为FileSystemStoragePathSource（位于sources / storage_path / file_system_storage_path_source *）。 FileSystemStoragePathSource监视特定的文件系统路径，监视数字子目录，并报告这些子目录的最新状态以作为其希望加载的版本。本文档介绍了FileSystemStoragePathSource的重要方面。您可能会发现方便地制作FileSystemStoragePathSource的副本，然后对其进行修改以适合您的需求。

首先，FileSystemStoragePathSource实现Source <StoragePath> API，这是Source <T> API的特化，其中T绑定到StoragePath。该API由单个方法SetAspiredVersionsCallback（）组成，该方法提供了一种闭包，源可以调用此闭包来传达其希望加载一组特定的可服务版本的信息。

FileSystemStoragePathSource以一种非常简单的方式使用aspired-versions回调：它定期检查文件系统（本质上是执行ls），并且如果找到一个或多个看起来像可服务版本的路径，它将确定哪个是最新版本并调用带有大小为1的列表的回调，仅包含该版本（在默认配置下）。因此，在任何给定时间，FileSystemStoragePathSource最多请求一个可加载的请求，并且其实现利用回调的幂等性使其自身保持无状态（使用相同的参数重复调用该回调不会有任何危害）。

FileSystemStoragePathSource具有一个静态初始化工厂（Create（）方法），该工厂带有配置协议消息。配置消息包括详细信息，例如要监视的基本路径和监视时间间隔。它还包括要发出的可服务流的名称。 （替代方法可能会从基本路径中提取可服务的流名称，以根据更深的目录层次结构发出多个可服务的流；这些变体超出了参考实现的范围。）

该实现的主要部分包括一个线程，该线程定期检查文件系统，以及一些用于识别和排序它发现的任何数字子路径的逻辑。该线程在SetAspiredVersionsCallback（）内部启动（不在Create（）中启动），因为这是源应“启动”并知道向何处发送有抱负版本请求的点。

使用您的Source加载TensorFlow会话
您可能希望将新的源模块与SavedModelBundleSourceAdapter（servables / tensorflow / saved_model_bundle_source_adapter *）结合使用，这会将源发出的每个路径解释为TensorFlow导出，并将每个路径转换为TensorFlow SavedModelBundle可服务对象的加载器。 您可能会将SavedModelBundle适配器插入AspiredVersionsManager中，该适配器负责实际加载和提供可服务对象。 在servables / tensorflow / simple_servers.cc中找到了将这三种模块链接在一起以获得工作服务器库的一个很好的例子。 这是主代码流程的逐步演练（错误处理错误；实际代码应格外小心）：

首先，创建一个经理：

```
std::unique_ptr<AspiredVersionsManager> manager = ...;
```

然后，创建一个SavedModelBundle源适配器并将其插入管理器：

```
std::unique_ptr<SavedModelBundleSourceAdapter> bundle_adapter;
SessionBundleSourceAdapterConfig config;
// ... populate 'config' with TensorFlow options.
TF_CHECK_OK(SavedModelBundleSourceAdapter::Create(config, &bundle_adapter));
ConnectSourceToTarget(bundle_adapter.get(), manager.get());
```

最后，创建路径源并将其插入SavedModelBundle适配器：

```
auto your_source = new YourPathSource(...);
ConnectSourceToTarget(your_source, bundle_adapter.get());
```

ConnectSourceToTarget（）函数（在core / target.h中定义）仅调用SetAspiredVersionsCallback（）将Source <T>连接到Target <T>（目标是捕获有向版本请求的模块，即适配器或管理器 ）。
