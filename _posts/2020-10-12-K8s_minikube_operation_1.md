---
layout: post
title:  "K8s minikube operation 1"
date:   2021-10-12 10:11:30
tags: [k8s]
---

    导语：
        K8s 基本操作系列之

## 总结


## 正文


## 问题
1. Exiting due to DRV_AS_ROOT: The "docker" driver should not be used with root privileges.

解决方法：
因为我是用root账号登录操作的。所以提示不能用root账号启动，得用别的账号。所以要创建一个新的账号进行操作，创建一个test账号进行启动。

```
adduser test
passwd test 设置密码
sudo usermod -aG docker $USER && newgrp docker 将tomx添加到docker组
su test 切换用户
```

2. X Exiting due to DRV_NOT_HEALTHY: Found driver(s) but none were healthy. See above for suggestions how to fix installed drivers.
提示错误
```
* minikube v1.23.2 on Centos 7.5.1804
* Unable to pick a default driver. Here is what was considered, in preference order:
  - docker: Not healthy: "docker version --format {{.Server.Os}}-{{.Server.Version}}" exit status 1: Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Get http://%2Fvar%2Frun%2Fdocker.sock/v1.37/version: dial unix /var/run/docker.sock: connect: permission denied
  - docker: Suggestion: Add your user to the 'docker' group: 'sudo usermod -aG docker $USER && newgrp docker' <https://docs.docker.com/engine/install/linux-postinstall/>
  - kvm2: Not healthy: libvirt group membership check failed:
user is not a member of the appropriate libvirt group
  - kvm2: Suggestion: Check that libvirtd is properly installed and that you are a member of the appropriate libvirt group (remember to relogin for group changes to take effect!) <https://minikube.sigs.k8s.io/docs/reference/drivers/kvm2/>
  - podman: Not installed: exec: "podman": executable file not found in $PATH
  - vmware: Not installed: exec: "docker-machine-driver-vmware": executable file not found in $PATH
  - virtualbox: Not installed: unable to find VBoxManage in $PATH

X Exiting due to DRV_NOT_HEALTHY: Found driver(s) but none were healthy. See above for suggestions how to fix installed drivers.
```
解决方法：
```
minikube start --driver=docker
```

3. 错误提示
```
[tomx@bbccfcca-7451-4863-a106-e4bb362f18be root]$ minikube start --driver=docker
* minikube v1.23.2 on Centos 7.5.1804
* Using the docker driver based on user configuration

X Exiting due to PROVIDER_DOCKER_NEWGRP: "docker version --format -" exit status 1: Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Get http://%2Fvar%2Frun%2Fdocker.sock/v1.37/version: dial unix /var/run/docker.sock: connect: permission denied
* Suggestion: Add your user to the 'docker' group: 'sudo usermod -aG docker $USER && newgrp docker'
* Documentation: https://docs.docker.com/engine/install/linux-postinstall/
```

解决办法：
```
sudo usermod -aG docker tomx && newgrp docker
```
$User 替换成指定的 user 名称

