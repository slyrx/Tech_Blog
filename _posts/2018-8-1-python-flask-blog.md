---
layout: post
title:  "Flask : build a Blog"
date:   2018-08-01 05:39:30
tags: [企业微信, python, Flask]
---

### 创建一个以Flask为基础的Blog
$ virtualenv blog

$ source blog/bin/activate
(blog) $ pip install Flask

mkdir app
touch app/{\__init\__,app,config,main,views}.py # 该命令行之间不能有空格。

文件树结构：
|__init__.py|Tells Python to use the app/ directory as a python package|
|app.py|The Flask app|
|config.py|Configuration variables for our Flask app|
|main.py|Entry-point for executing our application|
|views.py|URL routes and views for the app|

这个文件树的建立，只是把原来写在一个py中的内容，划分成多个模块了。
A circular import occurs when two modules mutually import each other and, hence, cannot be imported at all.
That is why we have broken our app into several modules and created a single entry-point that controls the ordering of imports.
(http://pcr54drkl.bkt.clouddn.com/Snip20180801_1.png)
(http://pcr54drkl.bkt.clouddn.com/Snip20180801_2.png)
(http://pcr54drkl.bkt.clouddn.com/Snip20180801_3.png)
(http://pcr54drkl.bkt.clouddn.com/Snip20180801_4.png)

文件树的import流程(http://pcr54drkl.bkt.clouddn.com/Snip20180801_5.png)



end
