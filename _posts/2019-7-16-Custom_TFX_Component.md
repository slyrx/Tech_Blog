---
layout: post
title:  "Custom TFX Component"
date:   2019-07-16 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第二十篇。
        本文档详细介绍了如何高级自定义 TFX 组件的作用。


## 目录
+ 自定义执行器或者自定义组件
+ 如何创建自定义组件
+ + 组件规格
+ + 执行器
+ + 组件接口
+ + 组装到TFX管道中
+ 部署自定义组件


## 正文
注意：本指南基于TFX 0.14.0，并且要求TFX> = 0.14.0。
自定义执行器或自定义组件
如果在组件的输入，输出和执行属性与现有组件相同的情况下仅需要自定义处理逻辑，则自定义执行程序就足够了。当任何输入，输出或执行属性与任何现有TFX组件不同时，需要一个自定义组件。

如何创建自定义组件？
开发自定义组件将需要：

新组件的一组定义的输入和输出工件规范。特别地，输入工件的类型应与产生工件的组件的输出工件类型一致，输出工件的类型应与消耗工件的组件的输入工件类型一致（如果有）。
新组件所需的非工件执行参数。
组件规格
ComponentSpec类通过定义组件的输入和输出工件以及将用于组件执行的参数来定义组件协定。其中包括三个部分：

输入：将被传递到组件执行器中的输入工件的规范。输入工件通常是上游组件的输出，因此具有相同的规格
输出：组件将产生的输出工件的规格。
参数：将传递到组件执行器中的执行属性的规范。这些是非工件参数，应在管道DSL中灵活定义并传递给新的组件实例。
这是ComponentSpec的示例，完整的示例可以在TFX GitHub存储库中找到。

```
class SlackComponentSpec(types.ComponentSpec):
  """ComponentSpec for Custom TFX Slack Component."""

  INPUTS = {
      'model_export': ChannelParameter(type=standard_artifacts.Model),
      'model_blessing': ChannelParameter(type=standard_artifacts.ModelBlessing),
  }
  OUTPUTS = {
      'slack_blessing': ChannelParameter(type=standard_artifacts.ModelBlessing),
  }
  PARAMETERS = {
      'slack_token': ExecutionParameter(type=Text),
      'slack_channel_id': ExecutionParameter(type=Text),
      'timeout_sec': ExecutionParameter(type=int),
  }
```

执行者
接下来，编写新组件的执行程序代码。 基本上，需要创建一个新的base_executor.BaseExecutor子类，并覆盖其Do函数。 在Do函数中，传入的参数input_dict，output_dict和exec_properties分别映射到ComponentSpec中定义的INPUTS，OUTPUTS和PARAMETERS。 对于exec_properties，可以直接通过字典查找来获取值。 对于input_dict和output_dict中的工件，可以使用方便的函数来获取工件的URI（在示例中，请参见model_export_uri和model_blessing_uri）或获取工件对象（在示例中，请参见slack_blessing）。

```

class Executor(base_executor.BaseExecutor):
  """Executor for Slack component."""
  ...
  def Do(self, input_dict: Dict[Text, List[types.TfxArtifact]],
         output_dict: Dict[Text, List[types.TfxArtifact]],
         exec_properties: Dict[Text, Any]) -> None:
    ...
    # Fetch execution properties from exec_properties dict.
    slack_token = exec_properties['slack_token']
    slack_channel_id = exec_properties['slack_channel_id']
    timeout_sec = exec_properties['timeout_sec']

    # Fetch input URIs from input_dict.
    model_export_uri = types.get_single_uri(input_dict['model_export'])
    model_blessing_uri = types.get_single_uri(input_dict['model_blessing'])

    # Fetch output artifact from output_dict.
    slack_blessing =
        types.get_single_instance(output_dict['slack_blessing'])
    ...
```

上面的示例仅显示了使用传入值的实现部分。请在TFX GitHub存储库中查看完整示例。

对自定义执行器进行单元测试
可以类似于此创建针对自定义执行程序的单元测试。

组件界面
现在最复杂的部分已经完成，下一步是将这些部分组装到组件接口中，以使组件可以在管道中使用。分几个步骤：

使组件接口成为base_component.BaseComponent的子类
用先前定义的ComponentSpec类分配一个类变量SPEC_CLASS
用先前定义的Executor类分配一个类变量EXECUTOR_SPEC
通过使用函数的参数来构造__init __（）构造函数，以构造ComponentSpec类的实例，并使用该值以及可选名称调用超函数。
创建组件实例时，将调用base_component.BaseComponent类中的类型检查逻辑，以确保传入的参数与ComponentSpec类中定义的类型信息兼容。

```
from slack_component import executor

class SlackComponent(base_component.BaseComponent):
  """Custom TFX Slack Component."""

  SPEC_CLASS = SlackComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(self,
               model_export: channel.Channel,
               model_blessing: channel.Channel,
               slack_token: Text,
               slack_channel_id: Text,
               timeout_sec: int,
               slack_blessing: Optional[channel.Channel] = None,
               name: Optional[Text] = None):
    slack_blessing = slack_blessing or channel.Channel(
        type_name='ModelBlessingPath',
        artifacts=[types.TfxArtifact('ModelBlessingPath')])
    spec = SlackComponentSpec(
        slack_token=slack_token,
        slack_channel_id=slack_channel_id,
        timeout_sec=timeout_sec,
        model=model_export,
        model_blessing=model_blessing,
        slack_blessing=slack_blessing)
    super(SlackComponent, self).__init__(spec=spec, name=name)
```

组装到TFX管道中
最后一步是将新的自定义组件插入TFX管道。 除了添加新组件的实例之外，还需要以下内容：

正确连接新组件的上游和下游组件。 这是通过引用新组件中上游组件的输出并引用下游组件中新组件的输出来完成的
构造管道时，将新的组件实例添加到组件列表中。
下面的示例突出显示了上述更改。 完整的示例可以在TFX GitHub存储库中找到。


```
def _create_pipeline():
  ...
  model_validator = ModelValidator(
      examples=example_gen.outputs['examples'], model=trainer.outputs['model'])

  slack_validator = SlackComponent(
      model=trainer.outputs['model'],
      model_blessing=model_validator.outputs['blessing'],
      slack_token=_slack_token,
      slack_channel_id=_slack_channel_id,
      timeout_sec=3600,
  )

  pusher = Pusher(
      ...
      model_blessing=slack_validator.outputs['slack_blessing'],
      ...)

  return pipeline.Pipeline(
      ...
      components=[
          ..., model_validator, slack_validator, pusher
      ],
      ...
  )
```

部署自定义组件
除了代码更改之外，还需要在管道运行环境中访问所有新添加的部分（ComponentSpec，Executor，组件接口），以便正确运行管道。