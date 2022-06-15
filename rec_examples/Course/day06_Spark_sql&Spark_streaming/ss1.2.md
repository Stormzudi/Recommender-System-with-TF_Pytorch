## 2、Spark Streaming编码实践

**Spark Streaming编码步骤：**

- 1，创建一个StreamingContext
- 2，从StreamingContext中创建一个数据对象
- 3，对数据对象进行Transformations操作
- 4，输出结果
- 5，开始和停止

**利用Spark Streaming实现WordCount**

需求：监听某个端口上的网络数据，实时统计出现的不同单词个数。

1，需要安装一个nc工具：sudo yum install -y nc

2，执行指令：nc -lk 9999 -v

```python
import os
# 配置spark driver和pyspark运行时，所使用的python解释器路径
PYSPARK_PYTHON = "/home/hadoop/miniconda3/envs/datapy365spark23/bin/python"
JAVA_HOME='/home/hadoop/app/jdk1.8.0_191'
SPARK_HOME = "/home/hadoop/app/spark-2.3.0-bin-2.6.0-cdh5.7.0"
# 当存在多个版本时，不指定很可能会导致出错
os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_PYTHON
os.environ['JAVA_HOME']=JAVA_HOME
os.environ["SPARK_HOME"] = SPARK_HOME

from pyspark import SparkContext
from pyspark.streaming import StreamingContext

if __name__ == "__main__":
    
    sc = SparkContext("local[2]",appName="NetworkWordCount")
    #参数2：指定执行计算的时间间隔
    ssc = StreamingContext(sc, 1)
    #监听ip，端口上的上的数据
    lines = ssc.socketTextStream('localhost',9999)
    #将数据按空格进行拆分为多个单词
    words = lines.flatMap(lambda line: line.split(" "))
    #将单词转换为(单词，1)的形式
    pairs = words.map(lambda word:(word,1))
    #统计单词个数
    wordCounts = pairs.reduceByKey(lambda x,y:x+y)
    #打印结果信息，会使得前面的transformation操作执行
    wordCounts.pprint()
    #启动StreamingContext
    ssc.start()
    #等待计算结束
    ssc.awaitTermination()
```

可视化查看效果：http://192.168.199.188:4040

点击streaming，查看效果

