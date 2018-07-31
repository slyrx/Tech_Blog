---
layout: post
title:  "Flask Framework Learning"
date:   2018-07-31 10:51:30
tags: [企业微信, python, Flask]
---

### 目标
创建一个动态的、数据驱动的 ***网站*** 和当代的 ***web app***

#### 目录分析
1. 创建第一个Flask
2. 创建关系数据库
3. 模版和视图
4. 表单和有效性验证
5. 用户授权
6. 构建管理仪表盘
7. AJAX和RESTful联动，知晓对应的api
8. 测试
9. 优秀的扩展
10. 应用部署

#### 概述
Flask is a lightweight Web framework written in Python.

Installing virtualenv
Virtualenv makes it easy to produce isolated Python environments, complete with their own copies of system and third-party packages.

Virtualenv solves a number of problems related to package management.
By using virtualenvs, you can create Python environments and install packages as a regular user.

#### 使用virtualenv
+ 基本的使用方法：$ virtualenv ENV
+ source命令的作用就是用来执行一个脚本。
+ 是一个新概念，大有裨益。

#### 对最简单的Flask App做解读
**from flask import Flask** # flask类引入一个独立的WSGI(Web Server Gateway Interface)应用。WSGI接口定义非常简单，它只要求Web开发者实现一个函数，就可以响应HTTP请求。
WSGI is the Python standard web server interface
**app = Flask(__name__)**

**@app.route('/')
   def index():
       return 'Hello, Flask!'**

**if __name__ == '__main__':
      app.run(debug=True)**

#### 一个博客项目的介绍
+ 这个项目会涉及到的功能
+ working with relational databases,
+ processing and validating form data,
+ (everyone's favorite) 可能是关于Ajax和RESTful的部分吧
+ testing



end
