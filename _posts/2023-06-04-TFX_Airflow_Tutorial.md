---
layout: post
title:  "TFX Airflow Tutorial"
date:   2023-06-04 10:11:30
tags: [tensorflow]
---

# Tutorial

## 环境要求
+ Linux/MacOS
+ Python 3.9 及更高版本
+ Virtualenv
+ Git


## 环境配置
```bash
    cd
    virtualenv -p python3 tfx-env
    source ~/tfx-env/bin/activate
    mkdir tfx; cd tfx



    git clone https://github.com/tensorflow/tfx.git
    cd ~/tfx/tfx/tfx/examples/workshop/setup
    ./setup_demo.sh
```

## 创建流水线框架

### 1. 打开一个新的终端窗口，在该窗口中执行下面的命令，启动 airflow 管理服务
```bash
    # Open a new terminal window, and in that window ...
    source ~/tfx-env/bin/activate
    cd
    airflow users  create --role Admin --username admin --email admin --firstname admin --lastname admin --password admin    
    airflow webserver -p 8080
```


### 2. 打开一个新的终端窗口，在该窗口中执行下面的命令，启动 airflow 调度服务
```bash
    # Open another new terminal window, and in that window ...
    source ~/tfx-env/bin/activate
    airflow scheduler
```

#### 在浏览器中：
打开浏览器，然后转到 http://127.0.0.1:8080

##### DAG 操作
登陆管理平台
![login page](https://www.tensorflow.org/static/tfx/tutorials/tfx/images/airflow_workshop/airflow-login.png)
运行 taxi 流水线
![DAG](https://www.tensorflow.org/static/tfx/tutorials/tfx/images/airflow_workshop/dag-home-full.png)

[![TensorFlow](https://www.tensorflow.org/static/tfx/tutorials/tfx/images/airflow_workshop/dag-buttons.png)](https://www.tensorflow.org/static/tfx/tutorials/tfx/images/airflow_workshop/dag-buttons.png)

![dag-button-refresh](https://www.tensorflow.org/static/tfx/tutorials/tfx/images/airflow_workshop/dag-button-refresh.png)

### 3. 打开一个新的终端窗口，在该窗口中执行下面的命令，打开对每个组件进行性能分析的 notebook
```bash
    # Open yet another new terminal window, and in that window ...
    # Assuming that you've cloned the TFX repo into ~/tfx
    source ~/tfx-env/bin/activate
    cd ~/tfx/tfx/tfx/examples/workshop/notebooks
    jupyter notebook
```


![login page](https://www.tensorflow.org/static/tfx/tutorials/tfx/images/airflow_workshop/notebook-ipynb.png)



