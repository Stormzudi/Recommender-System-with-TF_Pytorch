## spark-core RDD常用算子练习

课程目标

- 说出RDD的三类算子
- 掌握transformation和action算子的基本使用

### 3.1 RDD 常用操作

- RDD 支持两种类型的操作：
  - transformation
    - 从一个已经存在的数据集创建一个新的数据集
      - rdd a ----->transformation ----> rdd b
    - 比如， map就是一个transformation 操作，把数据集中的每一个元素传给一个函数并**返回一个新的RDD**，代表transformation操作的结果 
  - action
    - 获取对数据进行运算操作之后的结果
    - 比如， reduce 就是一个action操作，使用某个函数聚合RDD所有元素的操作，并**返回最终计算结果**

- 所有的transformation操作都是惰性的（lazy）
  - 不会立即计算结果
  - 只记下应用于数据集的transformation操作
  - 只有调用action一类的操作之后才会计算所有transformation
  - 这种设计使Spark运行效率更高
  - 例如map reduce 操作，map创建的数据集将用于reduce，map阶段的结果不会返回，仅会返回reduce结果。
- *persist* 操作
  - *persist*操作用于将数据缓存 可以缓存在内存中 也可以缓存到磁盘上， 也可以复制到磁盘的其它节点上

### 3.2 RDD Transformation算子

- map: map(func)

  - 将func函数作用到数据集的每一个元素上，生成一个新的RDD返回

  ``` shell
  >>> rdd1 = sc.parallelize([1,2,3,4,5,6,7,8,9],3)
  >>> rdd2 = rdd1.map(lambda x: x+1)
  >>> rdd2.collect()
  [2, 3, 4, 5, 6, 7, 8, 9, 10]
  ```

  ```shell
  >>> rdd1 = sc.parallelize([1,2,3,4,5,6,7,8,9],3)
  >>> def add(x):
  ...     return x+1
  ...
  >>> rdd2 = rdd1.map(add)
  >>> rdd2.collect()
  [2, 3, 4, 5, 6, 7, 8, 9, 10]
  ```

  

  ![](/img/rdd_map.png)

- filter

  - filter(func) 选出所有func返回值为true的元素，生成一个新的RDD返回

  ```shell
  >>> rdd1 = sc.parallelize([1,2,3,4,5,6,7,8,9],3)
  >>> rdd2 = rdd1.map(lambda x:x*2)
  >>> rdd3 = rdd2.filter(lambda x:x>4)
  >>> rdd3.collect()
  [6, 8, 10, 12, 14, 16, 18]
  ```

- flatmap

  - flatMap会先执行map的操作，再将所有对象合并为一个对象

  ```shell
  >>> rdd1 = sc.parallelize(["a b c","d e f","h i j"])
  >>> rdd2 = rdd1.flatMap(lambda x:x.split(" "))
  >>> rdd2.collect()
  ['a', 'b', 'c', 'd', 'e', 'f', 'h', 'i', 'j']
  ```

  - flatMap和map的区别：flatMap在map的基础上将结果合并到一个list中

  ```shell
  >>> rdd1 = sc.parallelize(["a b c","d e f","h i j"])
  >>> rdd2 = rdd1.map(lambda x:x.split(" "))
  >>> rdd2.collect()
  [['a', 'b', 'c'], ['d', 'e', 'f'], ['h', 'i', 'j']]
  ```

- union

  - 对两个RDD求并集

  ```shell
  >>> rdd1 = sc.parallelize([("a",1),("b",2)])
  >>> rdd2 = sc.parallelize([("c",1),("b",3)])
  >>> rdd3 = rdd1.union(rdd2)
  >>> rdd3.collect()
  [('a', 1), ('b', 2), ('c', 1), ('b', 3)]
  ```

- intersection

  - 对两个RDD求交集

  ```python
  >>> rdd1 = sc.parallelize([("a",1),("b",2)])
  >>> rdd2 = sc.parallelize([("c",1),("b",3)])
  >>> rdd3 = rdd1.union(rdd2)
  >>> rdd4 = rdd3.intersection(rdd2)
  >>> rdd4.collect()
  [('c', 1), ('b', 3)]
  ```

- groupByKey

  - 以元组中的第0个元素作为key，进行分组，返回一个新的RDD

  ```shell
  >>> rdd1 = sc.parallelize([("a",1),("b",2)])
  >>> rdd2 = sc.parallelize([("c",1),("b",3)])
  >>> rdd3 = rdd1.union(rdd2)
  >>> rdd4 = rdd3.groupByKey()
  >>> rdd4.collect()
  [('a', <pyspark.resultiterable.ResultIterable object at 0x7fba6a5e5898>), ('c', <pyspark.resultiterable.ResultIterable object at 0x7fba6a5e5518>), ('b', <pyspark.resultiterable.ResultIterable object at 0x7fba6a5e5f28>)]
  
  ```

  - groupByKey之后的结果中 value是一个Iterable

  ```python
  >>> result[2]
  ('b', <pyspark.resultiterable.ResultIterable object at 0x7fba6c18e518>)
  >>> result[2][1]
  <pyspark.resultiterable.ResultIterable object at 0x7fba6c18e518>
  >>> list(result[2][1])
  [2, 3]
  ```

  - reduceByKey

    - 将key相同的键值对，按照Function进行计算

    ```python
    >>> rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
    >>> rdd.reduceByKey(lambda x,y:x+y).collect()
    [('b', 1), ('a', 2)]
    ```

  - sortByKey

    - `sortByKey`(*ascending=True*, *numPartitions=None*, *keyfunc=<function RDD.<lambda>>*)

      Sorts this RDD, which is assumed to consist of (key, value) pairs.

    ```python
    >>> tmp = [('a', 1), ('b', 2), ('1', 3), ('d', 4), ('2', 5)]
    >>> sc.parallelize(tmp).sortByKey().first()
    ('1', 3)
    >>> sc.parallelize(tmp).sortByKey(True, 1).collect()
    [('1', 3), ('2', 5), ('a', 1), ('b', 2), ('d', 4)]
    >>> sc.parallelize(tmp).sortByKey(True, 2).collect()
    [('1', 3), ('2', 5), ('a', 1), ('b', 2), ('d', 4)]
    >>> tmp2 = [('Mary', 1), ('had', 2), ('a', 3), ('little', 4), ('lamb', 5)]
    >>> tmp2.extend([('whose', 6), ('fleece', 7), ('was', 8), ('white', 9)])
    >>> sc.parallelize(tmp2).sortByKey(True, 3, keyfunc=lambda k: k.lower()).collect()
    [('a', 3), ('fleece', 7), ('had', 2), ('lamb', 5),...('white', 9), ('whose', 6)]
    ```

    

### 3.3 RDD Action算子

- collect

  - 返回一个list，list中包含 RDD中的所有元素
  - 只有当数据量较小的时候使用Collect 因为所有的结果都会加载到内存中

- reduce

  - **reduce**将**RDD**中元素两两传递给输入函数，同时产生一个新的值，新产生的值与RDD中下一个元素再被传递给输入函数直到最后只有一个值为止。

  ```shell
  >>> rdd1 = sc.parallelize([1,2,3,4,5])
  >>> rdd1.reduce(lambda x,y : x+y)
  15
  ```

- first

  - 返回RDD的第一个元素

  ```python
  >>> sc.parallelize([2, 3, 4]).first()
  2
  ```

- take

  - 返回RDD的前N个元素
  - `take`(*num*)

  ``` shell
  >>> sc.parallelize([2, 3, 4, 5, 6]).take(2)
  [2, 3]
  >>> sc.parallelize([2, 3, 4, 5, 6]).take(10)
  [2, 3, 4, 5, 6]
  >>> sc.parallelize(range(100), 100).filter(lambda x: x > 90).take(3)
  [91, 92, 93]
  ```

- count

  返回RDD中元素的个数

  ```
  >>> sc.parallelize([2, 3, 4]).count()
  3
  ```

### 3.4 Spark RDD两类算子执行示意

![s5](/img/s5.png)

![s6](/img/s6.png)