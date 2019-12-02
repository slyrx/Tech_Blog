---
layout: post
title:  "Creating a new kind of servable"
date:   2019-07-03 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第七篇。
        本文档说明了如何使用一种新的可服务方式来扩展 TensorFlow Serving 。


## 目录
+ 为你的服务定义一个加载器和源适配器
+ 在管理器中对可服务对象进行排序
+ 访问已加载的你的服务对象
+ 高级：安排多个可服务实例共享状态

创建一种新的可食用
本文档说明了如何使用一种新的servable扩展TensorFlow Serving。最突出的可服务类型是SavedModelBundle，但定义其他类型的可服务类型，服务于模型附带的数据可能很有用。示例包括：词汇表，特征转换逻辑。任何C ++类都可以是可服务的，例如int，std :: map <string，int>或二进制文件中定义的任何类-我们将其称为YourServable。

为YourServable定义加载程序和SourceAdapter
要使TensorFlow Serving能够管理和服务YourServable，您需要定义两件事：

一个Loader类，用于加载，提供访问和卸载YourServable的实例。

一个SourceAdapter可以从某种基础数据格式（例如文件系统路径。作为SourceAdapter的替代方法，您可以编写完整的Source。但是，由于SourceAdapter方法更为通用和模块化，因此我们在这里重点介绍它。

在core / loader.h中定义了Loader抽象。它要求您定义用于加载，访问和卸载可服务类型的方法。加载servable的数据可以来自任何地方，但通常来自存储系统路径。让我们假设YourServable就是这种情况。让我们进一步假设您已经拥有一个满意的Source <StoragePath>（如果不满意，请参阅“定制源”文档）。

除了您的Loader，您还需要定义一个SourceAdapter，该SourceAdapter从给定的存储路径实例化一个Loader。大多数简单的用例都可以使用SimpleLoaderSourceAdapter类（位于core / simple_loader.h中）简洁地指定两个对象。高级用例可能会选择使用较低级别的API分别指定Loader和SourceAdapter类，例如SourceAdapter是否需要保留某些状态，和/或是否需要在Loader实例之间共享状态。

在servables / hashmap / hashmap_source_adapter.cc中有一个使用SimpleLoaderSourceAdapter的简单哈希图servable的参考实现。您可能会发现方便地制作HashmapSourceAdapter的副本，然后对其进行修改以适合您的需求。

HashmapSourceAdapter的实现包括两个部分：

从文件在LoadHashmapFromFile（）中加载哈希图的逻辑。

使用SimpleLoaderSourceAdapter定义基于LoadHashmapFromFile（）发出哈希映射加载程序的SourceAdapter。可以从HashmapSourceAdapterConfig类型的配置协议消息中实例化新的SourceAdapter。当前，配置消息仅包含文件格式，并且出于参考实现的目的，仅支持一种简单格式。

注意在析构函数中对Detach（）的调用。要求此调用以避免崩溃状态与其他线程中任何正在进行的Creator Lambda调用之间的竞争。 （尽管此简单的源适配器没有任何状态，但是基类仍然强制调用了Detach（）。）

安排将YourServable对象加载到管理器中
这是如何将新的SourceAdapter for YourServable加载程序连接到存储路径的基本源和管理器（错误处理错误；实际代码应更加小心）：

首先，创建一个经理：
```
std::unique_ptr<AspiredVersionsManager> manager = ...;
```

然后，创建YourServable源适配器并将其插入管理器：

```
auto your_adapter = new YourServableSourceAdapter(...);
ConnectSourceToTarget(your_adapter, manager.get());
```

最后，创建一个简单的路径源并将其插入适配器：
```
std::unique_ptr<FileSystemStoragePathSource> path_source;
// Here are some FileSystemStoragePathSource config settings that ought to get
// it working, but for details please see its documentation.
FileSystemStoragePathSourceConfig config;
// We just have a single servable stream. Call it "default".
config.set_servable_name("default");
config.set_base_path(FLAGS::base_path /* base path for our servable files */);
config.set_file_system_poll_wait_seconds(1);
TF_CHECK_OK(FileSystemStoragePathSource::Create(config, &path_source));
ConnectSourceToTarget(path_source.get(), your_adapter.get());
```

访问已加载的YourServable对象
以下是获取并使用已加载的YourServable的句柄的方法：

```
auto handle_request = serving::ServableRequest::Latest("default");
ServableHandle<YourServable*> servable;
Status status = manager->GetServableHandle(handle_request, &servable);
if (!status.ok()) {
  LOG(INFO) << "Zero versions of 'default' servable have been loaded so far";
  return;
}
// Use the servable.
(*servable)->SomeYourServableMethod();
```

高级：安排多个可服务实例共享状态
SourceAdapters可以容纳在多个发出的可服务对象之间共享的状态。 例如：

多个线程使用的共享线程池或其他资源。

多个可服务对象使用的共享只读数据结构，以避免在每个可服务实例中复制数据结构的时间和空间开销。

可以通过SourceAdapter迅速创建其初始化时间和大小可以忽略的共享状态（例如线程池），然后将共享状态嵌入到每个发出的可服务加载器中。 昂贵或大型共享状态的创建应推迟到第一个适用的Loader :: Load（）调用，即由管理者控制。 对称地，使用昂贵/大型共享状态对最终可服务对象的Loader :: Unload（）调用应将其删除。
