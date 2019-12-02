---
layout: post
title:  "Use TensorFlow Serving with Kubernetes"
date:   2019-07-02 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第六篇。
        本篇描述了如何使用在 Docker 容器中运行的 TensorFlow Serving 组件为TensorFlow ResNet 模型提供服务以及如何使用 Kubernetes 部署服务集群。

## 目录
+ Part 1: 安装
+ + 下载 ResNet 保存模型
+ Part 2: 在 Docker 中进行运行
+ + 为部署提交镜像
+ + 启动服务
+ + 查询服务
+ Part 3: 在 kubernetes 中进行部署
+ + 云服务项目登录
+ + 创建容器集群
+ + 上传 Docker 镜像
+ + 创建 Kubernetes 的部署和服务
+ + 查询模型

本教程展示了如何使用在Docker容器中运行的TensorFlow Serving组件为TensorFlow ResNet模型提供服务以及如何使用Kubernetes部署服务集群。

要了解有关TensorFlow Serving的更多信息，我们推荐TensorFlow Serving基本教程和TensorFlow Serving高级教程。

要了解有关TensorFlow ResNet模型的更多信息，建议阅读TensorFlow中的ResNet。

第1部分进行环境设置
第2部分展示了如何运行本地Docker服务映像
第3部分展示了如何在Kubernetes中进行部署。

第1部分：设置
在开始之前，请先安装Docker。

下载ResNet SavedModel
如果我们已有一个本地模型目录，请清除它：

```
rm -rf /tmp/resnet
```

深度残差网络（简称ResNets）提供了身份映射的突破性思想，以便能够训练非常深的卷积神经网络。 对于我们的示例，我们将为ImageNet数据集下载ResNet的TensorFlow SavedModel。

```
mkdir /tmp/resnet
curl -s http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NHWC_jpg.tar.gz | \
tar --strip-components=2 -C /tmp/resnet -xvz
```

我们可以验证是否具有SavedModel：

```
$ ls /tmp/resnet/*
saved_model.pb  variables
```

第2部分：在Docker中运行
提交映像以进行部署
现在，我们要获取一个服务映像，并将所有更改提交到新映像$ USER / resnet_serving中，以进行Kubernetes部署。

首先，我们将服务图片作为守护程序运行：

```
docker run -d --name serving_base tensorflow/serving
```

接下来，我们将ResNet模型数据复制到容器的模型文件夹中：

```
docker cp /tmp/resnet serving_base:/models/resnet
```

最后，我们将容器提交给ResNet模型服务：

```
docker commit --change "ENV MODEL_NAME resnet" serving_base \
  $USER/resnet_serving
```

现在，让我们停止投放基本容器

```
docker kill serving_base
docker rm serving_base
```

启动服务器
现在，让我们使用ResNet模型启动容器，以便准备好提供服务，并公开gRPC端口8500：

```
docker run -p 8500:8500 -t $USER/resnet_serving &
```

查询服务器
对于客户端，我们需要克隆TensorFlow Serving GitHub存储库：

```
git clone https://github.com/tensorflow/serving
cd serving
```

使用resnet_client_grpc.py查询服务器。客户端下载图像并通过gRPC发送图像，以分类为ImageNet类别。

```
tools/run_in_docker.sh python tensorflow_serving/example/resnet_client_grpc.py
```

这应该导致如下输出：

```
outputs {
  key: "classes"
  value {
    dtype: DT_INT64
    tensor_shape {
      dim {
        size: 1
      }
    }
    int64_val: 286
  }
}
outputs {
  key: "probabilities"
  value {
    dtype: DT_FLOAT
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 1001
      }
    }
    float_val: 2.41628322328e-06
    float_val: 1.90121829746e-06
    float_val: 2.72477100225e-05
    float_val: 4.42638565801e-07
    float_val: 8.98362372936e-07
    float_val: 6.84421956976e-06
    float_val: 1.66555237229e-05
...
    float_val: 1.59407863976e-06
    float_val: 1.2315689446e-06
    float_val: 1.17812135159e-06
    float_val: 1.46365800902e-05
    float_val: 5.81210713335e-07
    float_val: 6.59980651108e-05
    float_val: 0.00129527016543
  }
}
model_spec {
  name: "resnet"
  version {
    value: 1538687457
  }
  signature_name: "serving_default"
}
```

有用！服务器成功分类了猫图像！

第3部分：在Kubernetes中部署
在本节中，我们使用第0部分中构建的容器映像在Google Cloud Platform中使用Kubernetes部署服务集群。

GCloud项目登录
在这里，我们假设您已经创建并登录了一个名为tensorflow-serving的gcloud项目。

```
gcloud auth login --project tensorflow-serving
```

创建一个容器集群
首先，我们创建一个用于服务部署的Google Kubernetes Engine集群。

```
$ gcloud container clusters create resnet-serving-cluster --num-nodes 5
```

哪个应该输出类似

```
Creating cluster resnet-serving-cluster...done.
Created [https://container.googleapis.com/v1/projects/tensorflow-serving/zones/us-central1-f/clusters/resnet-serving-cluster].
kubeconfig entry generated for resnet-serving-cluster.
NAME                       ZONE           MASTER_VERSION  MASTER_IP        MACHINE_TYPE   NODE_VERSION  NUM_NODES  STATUS
resnet-serving-cluster  us-central1-f  1.1.8           104.197.163.119  n1-standard-1  1.1.8         5          RUNNING
```

为gcloud容器命令设置默认集群，并将集群凭证传递给kubectl。

```
gcloud config set container/cluster resnet-serving-cluster
gcloud container clusters get-credentials resnet-serving-cluster
```

这应导致：

```
Fetching cluster endpoint and auth data.
kubeconfig entry generated for resnet-serving-cluster.
```

上载Docker映像
现在，让我们将映像推送到Google Container Registry，以便我们可以在Google Cloud Platform上运行它。

首先，我们使用Container Registry格式和项目名称标记$ USER / resnet_serving图片，

```
docker tag $USER/resnet_serving gcr.io/tensorflow-serving/resnet
```

接下来，我们将图像推送到注册表，

```
gcloud docker -- push gcr.io/tensorflow-serving/resnet
```

创建Kubernetes部署和服务
该部署包含由Kubernetes部署控制的3个resnet_inference服务器副本。 Kubernetes服务与外部负载均衡器在外部公开副本。

我们使用示例Kubernetes配置resnet_k8s.yaml创建它们。

```
kubectl create -f tensorflow_serving/example/resnet_k8s.yaml
```

输出：

```
deployment "resnet-deployment" created
service "resnet-service" created
```

要查看部署和Pod的状态，请执行以下操作

```
$ kubectl get deployments
NAME                    DESIRED   CURRENT   UP-TO-DATE   AVAILABLE   AGE
resnet-deployment    3         3         3            3           5s
```

```
$ kubectl get pods
NAME                         READY     STATUS    RESTARTS   AGE
resnet-deployment-bbcbc   1/1       Running   0          10s
resnet-deployment-cj6l2   1/1       Running   0          10s
resnet-deployment-t1uep   1/1       Running   0          10s
```

要查看服务状态：

```
$ kubectl get services
NAME                    CLUSTER-IP       EXTERNAL-IP       PORT(S)     AGE
resnet-service       10.239.240.227   104.155.184.157   8500/TCP    1m
```

一切启动和运行可能需要一段时间。

```
$ kubectl describe service resnet-service
Name:           resnet-service
Namespace:      default
Labels:         run=resnet-service
Selector:       run=resnet-service
Type:           LoadBalancer
IP:         10.239.240.227
LoadBalancer Ingress:   104.155.184.157
Port:           <unset> 8500/TCP
NodePort:       <unset> 30334/TCP
Endpoints:      <none>
Session Affinity:   None
Events:
  FirstSeen LastSeen    Count   From            SubobjectPath   Type        Reason      Message
  --------- --------    -----   ----            -------------   --------    ------      -------
  1m        1m      1   {service-controller }           Normal      CreatingLoadBalancer    Creating load balancer
  1m        1m      1   {service-controller }           Normal      CreatedLoadBalancer Created load balancer
```

服务外部IP地址在LoadBalancer入口旁边列出。

查询模型
现在，我们可以从本地主机在其外部地址查询服务。

```
$ tools/run_in_docker.sh python \
  tensorflow_serving/example/resnet_client_grpc.py \
  --server=104.155.184.157:8500
outputs {
  key: "classes"
  value {
    dtype: DT_INT64
    tensor_shape {
      dim {
        size: 1
      }
    }
    int64_val: 286
  }
}
outputs {
  key: "probabilities"
  value {
    dtype: DT_FLOAT
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 1001
      }
    }
    float_val: 2.41628322328e-06
    float_val: 1.90121829746e-06
    float_val: 2.72477100225e-05
    float_val: 4.42638565801e-07
    float_val: 8.98362372936e-07
    float_val: 6.84421956976e-06
    float_val: 1.66555237229e-05
...
    float_val: 1.59407863976e-06
    float_val: 1.2315689446e-06
    float_val: 1.17812135159e-06
    float_val: 1.46365800902e-05
    float_val: 5.81210713335e-07
    float_val: 6.59980651108e-05
    float_val: 0.00129527016543
  }
}
model_spec {
  name: "resnet"
  version {
    value: 1538687457
  }
  signature_name: "serving_default"
}
```

您已经成功在Kubernetes中部署了ResNet模型作为服务！
