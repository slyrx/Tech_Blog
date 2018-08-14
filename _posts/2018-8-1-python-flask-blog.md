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
+
A circular import occurs when two modules mutually import each other and, hence, cannot be imported at all.
That is why we have broken our app into several modules and created a single entry-point that controls the ordering of imports.

![config.py](http://pcr54drkl.bkt.clouddn.com/Snip20180801_1.png)
![app.py](http://pcr54drkl.bkt.clouddn.com/Snip20180801_2.png)
![views.py](http://pcr54drkl.bkt.clouddn.com/Snip20180801_3.png)
![main.py](http://pcr54drkl.bkt.clouddn.com/Snip20180801_4.png)

文件树的import流程

![文件树的import流程](http://pcr54drkl.bkt.clouddn.com/Snip20180801_5.png)

#### 添加关系数据库
+ SQLAlchemy is an extremely powerful library for working with relational databases in Python.
+ Instead of writing SQL queries by hand, we can use normal Python objects to represent database tables and execute queries.
+ SQLAlchemy supports a multitude of popular database dialects, including SQLite, MySQL, and PostgreSQL.

#### Flask扩展库
http:// flask.pocoo.org/extensions/

#### 新命令
touch 主要功能是：改变timestamps;新建空白文件

+ **IPython**, a sophisticated shell with features such as tab-completion (that the default Python shell lacks).
+ In IPython, you can use an underscore (\_) to reference the return- value of the previous line.

2018年8月1日 星期三
Adding Flask-Migrate to our project

2018年8月4日 星期六
Jinja2 is a fast, flexible, and secure templating engine.
关于库引入的概念更深一步了，关键在sys.path

2018年8月5日 星期日
Deleting entries

2018年8月13日 星期一
Serving static files

2018年8月15日 星期三
Customizing Admin model forms
end
