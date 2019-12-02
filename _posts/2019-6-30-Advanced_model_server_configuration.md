---
layout: post
title:  "Advanced model server configuration - Tensorflow Serving Configuration"
date:   2019-06-30 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第四篇。
        本篇描述了将介绍 tensorflow 服务的众多配置内容。

## 目录
+ 概览
+ 模型服务配置
+ + 重新加载模型服务配置
+ + 模型服务配置细节
+ + 配置一个模型
+ + 服务一个指定版本的模型
+ + 服务多个指定版本的模型
+ + 为模型版本分配字符串标签，以简化告密和回滚。
+ 监控配置
+ 批处理配置
+ 杂项标志


## 正文
Tensorflow服务配置
在本指南中，我们将介绍Tensorflow服务的众多配置点。

总览
尽管大多数配置与Model Server相关，但是有许多方法可以指定Tensorflow服务的行为：

模型服务器配置：指定模型名称，路径，版本策略和标签，日志记录配置等
监视配置：启用和配置Prometheus监视
批处理配置：启用批处理并配置其参数
杂项标志：许多杂项。可以提供标记来微调Tensorflow Serving部署的行为
模型服务器配置
提供模型的最简单方法是提供--model_name和--model_base_path标志（或者，如果使用Docker，则设置MODEL_NAME环境变量）。但是，如果您想提供多个模型，或者配置新版本的轮询频率之类的选项，则可以通过编写模型服务器配置文件来实现。

您可以使用--model_config_file标志提供此配置文件，并通过设置--model_config_file_poll_wait_seconds标志来指示Tensorflow Serving在指定的路径上定期轮询此配置文件的更新版本。

使用Docker的示例：

docker run -t --rm -p 8501：8501 \
    -v“ $（pwd）/ models /：/ models /” tensorflow /serving\
    --model_config_file = / models / models.config \
    --model_config_file_poll_wait_seconds = 60

重新加载模型服务器配置
有两种方法可以重新加载模型服务器配置：

通过设置--model_config_file_poll_wait_seconds标志来指示服务器定期在--model_config_file文件路径中检查新的配置文件。

通过向服务器发出HandleReloadConfigRequest RPC调用并以编程方式提供新的Model Server配置。

请注意，每次服务器加载新的配置文件时，它将用于实现新指定配置的内容，并且仅实现新指定配置的内容。这意味着，如果第一个配置文件中存在模型A（该模型被仅包含模型B的文件替换），则服务器将加载模型B并卸载模型A。

模型服务器配置详细信息
提供的模型服务器配置文件必须是ASCII ModelServerConfig协议缓冲区。请参阅以下内容以了解ASCII协议缓冲区的外观。

对于除最高级用例以外的所有用例，您都想使用ModelConfigList选项，该选项是ModelConfig协议缓冲区的列表。这是一个基本示例，然后我们将介绍以下高级选项。

model_config_list {
  配置{
    名称：“ my_first_model”
    base_path：'/ tmp / my_first_model /'
  }
  配置{
    名称：“ my_second_model”
    base_path：'/ tmp / my_second_model /'
  }
}

配置一个模型
每个ModelConfig指定一个要服务的模型，包括其名称和模型服务器应在其中寻找要服务的模型版本的路径，如上例所示。默认情况下，服务器将提供具有最大版本号的版本。可以通过更改model_version_policy字段来覆盖此默认值。

提供模型的特定版本
要提供特定版本的模型，而不必总是过渡到具有最大版本号的模型，请将model_version_policy设置为“ specific”并提供您要提供的版本号。例如，将版本42固定为可投放的版本：

model_version_policy {
  具体{
    版本：42
  }
}

如果发现最新版本存在问题，此选项对于回滚到已知良好版本很有用。

提供模型的多个版本
同时提供模型的多个版本，例如要启用包含少量流量的暂定新版本，请将model_version_policy设置为“ specific”并提供多个版本号。例如，要提供版本42和43：

model_version_policy {
  具体{
    版本：42
    版本：43
  }
}

为模型版本分配字符串标签，以简化Canary和Rollback
有时将间接级别添加到模型版本会很有帮助。不必让所有客户都知道他们应查询版本42，您可以为每个客户当前应查询的版本分配一个别名，例如“ stable”。如果要将流量的一部分重定向到暂定的Canary模型版本，则可以使用第二个别名“ canary”。

您可以配置这些模型版本别名或标签，如下所示：

model_version_policy {
  具体{
    版本：42
    版本：43
  }
}
version_labels {
  关键：“稳定”
  价值：42
}
version_labels {
  关键：“金丝雀”
  价值：43
}

在上面的示例中，您正在提供版本42和43，并将标签“ stable”与版本42相关联，并将标签“ canary”与版本43相关联。您可以让客户将查询直接定向到“ stable”或“ canary”之一（也许基于对用户ID进行哈希处理），使用ModelSpec协议缓冲区的version_label字段，然后在不通知客户端的情况下在服务器上向前移动标签。完成Canarying版本43并准备将其升级到稳定版本后，可以将配置更新为：

model_version_policy {
  具体{
    版本：42
    版本：43
  }
}
version_labels {
  关键：“稳定”
  价值：43
}
version_labels {
  关键：“金丝雀”
  价值：43
}

如果随后需要执行回滚，则可以还原为版本42为“稳定”的旧配置。否则，您可以前进，方法是卸载版本42，并在准备好新版本44时加载它，然后将Canary标签前进到44，依此类推。

请注意，标签只能分配给已经加载并可以投放的模型版本。一旦有可用的模型版本，就可以即时重新加载模型配置以为其分配标签。如上所述，这可以使用HandleReloadConfigRequest RPC来实现，或者如果将服务器设置为定期轮询文件系统以获取配置文件，则可以实现此目的。

如果要为尚未加载的版本分配标签（例如，通过在启动时同时提供模型版本和标签），则必须将--alow_version_labels_for_unavailable_models标志设置为true，以允许使用新标签分配给尚未加载的模型版本。

请注意，这仅适用于新版本标签（即当前未分配给版本的标签）。这是为了确保在版本交换期间，服务器不会过早地将标签分配给新版本，从而在加载新版本时丢弃所有发往该标签的请求。

为了遵守此安全检查，如果重新分配一个已在使用的版本标签，则必须仅将其分配给已加载的版本。例如，如果要将标签从指向版本N移到版本N + 1，则可以首先提交包含版本N和N + 1的配置，然后提交包含版本N + 1（标签）的配置。指向N + 1，没有版本N。

监控配置
您可以使用--monitoring_config_file标志为服务器提供监视配置，以指定包含MonitoringConfig协议缓冲区的文件。这是一个例子：

prometheus_config {
  启用：是，
  路径：“ / monitoring / prometheus / metrics”
}

要从上述监视URL读取指标，您首先需要通过设置--rest_api_port标志来启用HTTP服务器。然后，您可以通过将--rest_api_port和path的值传递给Prometheus Server，以从Model Server中获取指标。

Tensorflow Serving收集Serving以及核心Tensorflow捕获的所有指标。

批处理配置
Model Server可以在各种设置中批量处理请求，以实现更好的吞吐量。对于服务器上的所有模型和模型版本，将在全局范围内完成此批处理的调度，以确保无论服务器当前正在服务多少个模型或模型版本，都可以最大程度地利用基础资源（更多详细信息）。您可以通过设置--enable_batching标志来启用此行为，并通过将配置传递给--batching_parameters_file标志来对其进行控制。

批处理参数文件示例：

max_batch_size {值：128}
batch_timeout_micros {值：0}
max_enqueued_batches {值：1000000}
num_batch_threads {值：8}

请参考批处理指南进行更深入的讨论，并参考有关参数的部分以了解如何设置参数。

杂项标志
除了指南中到目前为止涵盖的标志之外，我们在这里还列出了其他一些值得注意的标志。有关完整列表，请参考源代码。

--port：用于监听gRPC API的端口
--rest_api_port：用于侦听HTTP / REST API的端口
--rest_api_timeout_in_ms：HTTP / REST API调用超时
--file_system_poll_wait_seconds：服务器在每个模型各自的model_base_path上轮询文件系统以获取新模型版本的时间
--enable_model_warmup：使用assets.extra /目录中用户提供的PredictionLogs启用模型预热