---
layout: post
title:  "Transform library for non-TFX users"
date:   2019-07-23 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第二十七篇。
        本文档介绍转换库的情况。


转换也可以作为独立库使用。 大多数库文档与TFX用户无关，因为TFX用户仅构造preprocessing_fn，其余的Transform库调用则由Transform组件进行。