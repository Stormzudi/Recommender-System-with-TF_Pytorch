## spark-core概述

课程目标：

- 知道RDD的概念
- 独立实现RDD的创建

### 2.1 什么是RDD

- RDD（Resilient Distributed Dataset）叫做弹性分布式数据集，是Spark中最基本的数据抽象，它代表一个不可变、可分区、里面的元素可并行计算的集合.
  - Dataset:一个数据集，简单的理解为集合，用于存放数据的
  - Distributed：它的数据是分布式存储，并且可以做分布式的计算
  - Resilient：弹性的
    - 它表示的是数据可以保存在磁盘，也可以保存在内存中
    - 数据分布式也是弹性的
    - 弹性:并不是指他可以动态扩展，而是容错机制。
      - RDD会在多个节点上存储，就和hdfs的分布式道理是一样的。hdfs文件被切分为多个block存储在各个节点上，而RDD是被切分为多个partition。不同的partition可能在不同的节点上
      - spark读取hdfs的场景下，spark把hdfs的block读到内存就会抽象为spark的partition。
      - spark计算结束，一般会把数据做持久化到hive，hbase，hdfs等等。我们就拿hdfs举例，将RDD持久化到hdfs上，RDD的每个partition就会存成一个文件，如果文件小于128M，就可以理解为一个partition对应hdfs的一个block。反之，如果大于128M，就会被且分为多个block，这样，一个partition就会对应多个block。
  - 不可变
  - 可分区
  - 并行计算

### 2.2 RDD的创建

- 第一步 创建sparkContext

  - SparkContext, Spark程序的入口. SparkContext代表了和Spark集群的链接, 在Spark集群中通过SparkContext来创建RDD
  - SparkConf  创建SparkContext的时候需要一个SparkConf， 用来传递Spark应用的基本信息

  ``` python
  conf = SparkConf().setAppName(appName).setMaster(master)
  sc = SparkContext(conf=conf)
  ```

- 创建RDD

  - 进入pyspark环境

  ```shell
  [hadoop@hadoop000 ~]$ pyspark
  Python 3.5.0 (default, Nov 13 2018, 15:43:53)
  [GCC 4.8.5 20150623 (Red Hat 4.8.5-28)] on linux
  Type "help", "copyright", "credits" or "license" for more information.
  19/03/08 12:19:55 WARN util.NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
  Setting default log level to "WARN".
  To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
  Welcome to
        ____              __
       / __/__  ___ _____/ /__
      _\ \/ _ \/ _ `/ __/  '_/
     /__ / .__/\_,_/_/ /_/\_\   version 2.3.0
        /_/
  
  Using Python version 3.5.0 (default, Nov 13 2018 15:43:53)
  SparkSession available as 'spark'.
  >>> sc
  <SparkContext master=local[*] appName=PySparkShell>
  ```

  - 在spark shell中 已经为我们创建好了 SparkContext 通过sc直接使用
  - 可以在spark UI中看到当前的Spark作业 在浏览器访问当前centos的4040端口

  ![](/img/sparkui.png)
  - Parallelized Collections方式创建RDD

    - 调用`SparkContext`的 `parallelize` 方法并且传入已有的可迭代对象或者集合

    ```python
    data = [1, 2, 3, 4, 5]
    distData = sc.parallelize(data)
    ```

    ``` shell
    >>> data = [1, 2, 3, 4, 5]
    >>> distData = sc.parallelize(data)
    >>> data
    [1, 2, 3, 4, 5]
    >>> distData
    ParallelCollectionRDD[0] at parallelize at PythonRDD.scala:175
    ```

    - 在spark ui中观察执行情况

    ![createrdd](/img/createrdd.png)

    - 在通过`parallelize`方法创建RDD 的时候可以指定分区数量

    ```shell
    >>> distData = sc.parallelize(data,5)
    >>> distData.reduce(lambda a, b: a + b)
    15
    ```

    - 在spark ui中观察执行情况

    ![](/img/createrdd2.png)

    -  Spark将为群集的每个分区（partition）运行一个任务（task）。 通常，可以根据CPU核心数量指定分区数量（每个CPU有2-4个分区）如未指定分区数量，Spark会自动设置分区数。

  - 通过外部数据创建RDD

    - PySpark可以从Hadoop支持的任何存储源创建RDD，包括本地文件系统，HDFS，Cassandra，HBase，Amazon S3等
    - 支持整个目录、多文件、通配符
    - 支持压缩文件

    ```shell
    >>> rdd1 = sc.textFile('file:///home/hadoop/tmp/word.txt')
    >>> rdd1.collect()
    ['foo foo quux labs foo bar quux abc bar see you by test welcome test', 'abc labs foo me python hadoop ab ac bc bec python']
    ```

    