
## How To Use
when uploading some change, you should wait for a little while, in order to let github refresh the jekyll server.

## issues
+ sometimes when you change the setting in **\_config.yml** , the page didn't show ang change. At this time, you should wait, or change some other True/False Setting to triggle the change updating.


## using docker as the test envirment
+ first get image
1. run docker app
2. docker pull docker pull jekyll/jekyll:3.8
3. docker run --name=jk -p 4000:4000 -it -v /Users/slyrx/slyrxStudio/github_good_projects/Tech_Blog:/srv/jekyll jekyll/jekyll:3.8 sh

+ rerun image
+ 开启 docker 必须在创建的位置执行，其它地方不认识
1. run docker app
2. docker ps -a
3. docker start -i 9949204b531a

## jeklly run
1. bundle update
2. bundle clean  
3. bundle exe jekyll build
4. bundle exec jekyll serve -H 0.0.0.0 -P 4000 -I


## Tensorflow 翻译进度

|名称|完成情况|
|---|---|
|[Guide]||
|[**Components**]||
|  [ExampleGen]||
|  [StatisticsGen]||
|  [SchemaGen]||
|  [ExampleValidator]||
|  [Transform]||
|  [Trainer]||
|  [Evaluator]||
|  [ModelValidator]||
|  [Pusher]||
|  [Advanced: Custom Components]||
|[**Orchestrators** 协调器]||
|  [Apache Airflow]||
|  [Apache Beam]||
|  [Kubeflow Pipelines]||
|[**Libraries**]||
|  [**Data validation**]||
|  [check and analyze data]||
|  [install]||
|  [get started]||
|[**Transform**]||
|  [Preprocess and transform data]||
|  [install]||
|  [get started]||
|[**Modeling for TFX**]||
|  [Design modeling code]||
|[**Model Analysis**]||
|  [install]||
|  [get started]||
|  [architecture]||
|  [improving Model Quality]||
|[**Serving**]||
|[Advanced model server Configuration](https://www.tensorflow.org/tfx/serving/custom_servable)||
|[Building Standard TensorFlow ModelServer](https://www.tensorflow.org/tfx/serving/serving_advanced)|✅|
|[Use TensorFlow Serving with Kubernetes](https://www.tensorflow.org/tfx/serving/serving_kubernetes)||
|[Creating a new kind of servable](https://www.tensorflow.org/tfx/serving/custom_servable)||
|[Creating a module that discovers new servable paths](https://www.tensorflow.org/tfx/serving/custom_source)||
|[Serving TensorFlow models with custom ops](https://www.tensorflow.org/tfx/serving/custom_ops)||
|[SignatureDefs in SavedModel for TensorFlow Serving](https://www.tensorflow.org/tfx/serving/signature_defs)||
|[**Related Projects**]||
|[Apache Beam]||
|[ML Metadata]||
|[TensorBoard]||

####################################################

|完成状态|名称|
|---|---|
||Guide|
||Components|
||ExampleGen|
||StatisticsGen|
||SchemaGen|
||ExampleValidator|
||Transform|
||Trainer|
||Evaluator|
||ModelValidator|
||Pusher|
||Advanced: Custom Components|
||Orchestrators|
||Apache Airflow|
||Apache Beam|
||Kubeflow Pipelines|
||Libraries|
||Data Validation|
|||
||Check and analyze data|
||Install|
||Get started|
||Transform|
|||
||Preprocess and transform data|
||Install|
||Get started|
||Modeling for TFX|
|||
||Design modeling code|
||Model Analysis|
|||
||Install|
||Get started|
||Architecture|
||Improving Model Quality|
||Using Fairness Indicators|
||Serving|
|||
||Serving models|
||TensorFlow Serving with Docker|
||Installation|
||Serve a TensorFlow model|
||Architecture|
||Advanced model server configuration|
||Build a TensorFlow ModelServer|
||Use TensorFlow Serving with Kubernetes|
||Create a new kind of servable|
||Create a module that discovers new servable paths|
||Serving TensorFlow models with custom ops|
||SignatureDefs in SavedModel for TensorFlow Serving|
||Related projects|
||Apache Beam|
||ML Metadata|
||TensorBoard|
