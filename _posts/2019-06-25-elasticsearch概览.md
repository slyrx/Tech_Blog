---
layout: post
title:  "ElasticSearch 概览"
date:   2019-06-20 10:11:30
tags: [ElasticSearch]
---

最近对 ElasticSearch 做了深入的研究，不得不说在搜索的领域，又打开了一扇有趣的大门。 ElasticSearch 是一个基于 Apache Lucene 的开源搜索引擎。而 Lucene 则是当今世界上最好的搜索引擎库，没有之一。但是, Lucene 的弊端在于它是一个库，入门门槛高，需要深入的了解检索的相关知识，才能驾驭它。ElasticSearch 的出现就是为了通过简单的 RESTful API 来隐藏 Lucene 的复杂性，从而让全文搜索变得简单。尽管，ElasticSearch 也是由 Java 编写的，但是在调用 ElasticSearch 时完全不用考虑语言的要求，任何编程语言都可以启动对 ElasticSearch 的调用。

这篇文章先对 ElasticSearch 进行概览，纵观 ElasticSearch 的基本原理，之后再针对高级功能进行深入研究。

## 基本设计
首先，从设计的角度出发，可以从逻辑和物理两个方面理解 ElasticSearch 。
### 设计出发点--逻辑
从逻辑设计出发，ElasticSearch 有3个基本概念：文档、类型和索引。文档是 ElasticSearch 里的基本单位；索引是一个容器；类型是容器中的一个表，它能囊括文档，同时也会被包含于索引中。<br>
这里强调一下索引的概念，什么是索引？就是将文档添加入库时的一个编号，以方便后续的操作。允许现有索引，但是还没有内容的情况。
### 设计出发点--物理
从物理设计出发， ElasticSearch 也有3个基本概念：节点、主分片和副本分片。节点就是一个 ElasticSearch 实例，也就是一个 ElasticSearch 进程，同一台服务器上开启多个 ElasticSearch 进程，就是开启多个节点；分片是 ElasticSearch 处理的最小单元，它里面存放的即是索引：分片又包括主分片和副本分片，副本分片是主分片的完整备份，用于**搜索**；主分片则是用于执行**索引**文档。
### 基本使用
基本的使用包括两个部分：1⃣️对新数据的索引；2⃣️对已索引好的数据进行搜索
### 索引新数据
ElasticSearch 通过 RESTful API 自动对文档进行索引。也就是，执行一条如下代码的语句，即执行完毕建立索引的过程：
```
curl -XPUT 'localhost:9200/get-together/group/1?pretty' -d '{"name": "ElasticSearch Denver", "organizer": "Lee"}'
```

执行结果会返回如下内容：
```
{
    "_index": "get-together",
    "_type": "group",
    "_id": "1",
    "_version": 1,
    "created": true
}
```
### 搜索并获取数据
搜索数据也是使用 RESTful API ，只是参数不同。在搜索部分，重点关注3个方面的内容：
1. 在哪里搜索
包括4种位置
+ 特定类型和索引中
+ 同一索引的多个字段
+ 多个索引
+ 所有索引
2. 回复的内容
包括4个方面
+ 请求耗时及超时情况，单位是毫秒
+ 查询了多少分片
+ 所有匹配文档统计
统计处需要注意的是，两个分片拥有的内容是相同的，所有在匹配的结果显示两个分片匹配到了；但因为是一样的内容，所以可显示的命中文档内容是只有一个。
+ 结果数组

3. 搜索什么以及如何搜索
基本的搜索语句，格式如下：
```
curl 'localhost:9200/get-together/group/_search?pretty' -d '{
    "query": {
        "query_string":{
            "query": "elasticsearch"
        }
    }
}'
```

在搜索时，涉及到一种方式称为“聚集”，这个方式相当于数据结构中的 Counter 操作。
一般地，在选择什么样的时机来进行聚集还是其它搜索方式时，依据以下的原则：
|对需求的了解情况|执行的操作行为|
|---|---|
|不清楚|聚集钻取|
|清楚|其它搜索|

同样是搜索的情况下，直接的 ID 检索也比普通的搜索更快。类比数组取值和 map 取值，map 最终还是有一个查询过程，而数组则是 O(1) 的时间复杂度。

## 集群管理






