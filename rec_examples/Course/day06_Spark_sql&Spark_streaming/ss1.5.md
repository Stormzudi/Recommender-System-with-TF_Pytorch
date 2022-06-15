## 5、Spark Streaming对接flume

flume作为日志实时采集的框架，可以与SparkStreaming实时处理框架进行对接，flume实时产生数据，sparkStreaming做实时处理。

Spark Streaming对接FlumeNG有两种方式，一种是FlumeNG将消息**Push**推给Spark Streaming，还有一种是Spark Streaming从flume 中**Pull**拉取数据。

### 5.1 Pull方式

- 1，安装flume1.6以上

- 2，下载依赖包

  spark-streaming-flume-assembly_2.11-2.3.0.jar放入到flume的lib目录下

- 3，写flume的agent，注意既然是拉取的方式，那么flume向自己所在的机器上产数据就行

- 4，编写flume-pull.conf配置文件

  ```properties
  simple-agent.sources = netcat-source
  simple-agent.sinks = spark-sink
  simple-agent.channels = memory-channel
   
  # source
  simple-agent.sources.netcat-source.type = netcat
  simple-agent.sources.netcat-source.bind = localhost
  simple-agent.sources.netcat-source.port = 44444
  
  # Describe the sink
  simple-agent.sinks.spark-sink.type = org.apache.spark.streaming.flume.sink.SparkSink
  simple-agent.sinks.spark-sink.hostname = localhost
  simple-agent.sinks.spark-sink.port = 41414
   
  # Use a channel which buffers events in memory
  simple-agent.channels.memory-channel.type = memory
   
  # Bind the source and sink to the channel
  simple-agent.sources.netcat-source.channels = memory-channel
  simple-agent.sinks.spark-sink.channel=memory-channel
  ```

- 5，启动flume

  flume-ng agent -n simple-agent -f flume-pull.conf -Dflume.root.logger=INFO,console

- 6，编写word count代码

  代码：

```python
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.flume import FlumeUtils

sc=SparkContext("local[2]","FlumeWordCount_Pull")
#处理时间间隔为2s
ssc=StreamingContext(sc,2)

#利用flume工具类创建pull方式的流
lines = FlumeUtils.createPollingStream(ssc, [("localhost",41414)])

lines1=lines.map(lambda x:x[1])
counts = lines1.flatMap(lambda line:line.split(" "))\
        .map(lambda word:(word,1))\
        .reduceByKey(lambda a,b:a+b)
counts.pprint()
#启动spark streaming应用
ssc.start()
#等待计算终止
ssc.awaitTermination()
```

启动

`bin/spark-submit --jars xxx/spark-streaming-flume-assembly_2.11-2.3.0.jar xxx/flume_pull.py`

## 5.2 push方式

大部分操作和之前一致

flume配置

```properties
simple-agent.sources = netcat-source
simple-agent.sinks = avro-sink
simple-agent.channels = memory-channel

simple-agent.sources.netcat-source.type = netcat
simple-agent.sources.netcat-source.bind = localhost
simple-agent.sources.netcat-source.port = 44444

simple-agent.sinks.avro-sink.type = avro
simple-agent.sinks.avro-sink.hostname = localhost
simple-agent.sinks.avro-sink.port = 41414
simple-agent.channels.memory-channel.type = memory
simple-agent.sources.netcat-source.channels = memory-channel

simple-agent.sources.netcat-source.channels = memory-channel
simple-agent.sinks.avro-sink.channel=memory-channel
```

代码：

```python
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.flume import FlumeUtils

sc=SparkContext("local[2]","FlumeWordCount_Push")
#处理时间间隔为2s
ssc=StreamingContext(sc,2)
#创建push方式的DStream
lines = FlumeUtils.createStream(ssc, "localhost",41414)
lines1=lines.map(lambda x:x[1].strip())
#对1s内收到的字符串进行分割
words=lines1.flatMap(lambda line:line.split(" "))
#映射为（word，1）元祖
pairs=words.map(lambda word:(word,1))
wordcounts=pairs.reduceByKey(lambda x,y:x+y)
wordcounts.pprint()
#启动spark streaming应用
ssc.start()
#等待计算终止
ssc.awaitTermination()
```