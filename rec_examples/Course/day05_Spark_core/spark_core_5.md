## spark-core实战

课程目标

- 独立实现spark standalone模式的启动
- 说出广播变量的概念

### 5.1通过spark实现ip地址查询

**需求**

在互联网中，我们经常会见到城市热点图这样的报表数据，例如在百度统计中，会统计今年的热门旅游城市、热门报考学校等，会将这样的信息显示在热点图中。

因此，我们需要通过日志信息（运行商或者网站自己生成）和城市ip段信息来判断用户的ip段，统计热点经纬度。

**ip日志信息**

在ip日志信息中，我们只需要关心ip这一个维度就可以了，其他的不做介绍

**思路**

1、 加载城市ip段信息，获取ip起始数字和结束数字，经度，纬度

2、 加载日志数据，获取ip信息，然后转换为数字，和ip段比较

3、 比较的时候采用二分法查找，找到对应的经度和纬度

4，对相同的经度和纬度做累计求和

**启动Spark集群**

- 进入到$SPARK_HOME/sbin目录

  - 启动Master	

  ```shell
  ./start-master.sh -h 192.168.199.188
  ```

  - 启动Slave

  ```shell
   ./start-slave.sh spark://192.168.199.188:7077
  ```

  - jps查看进程

  ```shell
  27073 Master
  27151 Worker
  ```

  - 关闭防火墙

  ```shell
  systemctl stop firewalld
  ```

  - 通过SPARK WEB UI查看Spark集群及Spark
    - http://192.168.199.188:8080/  监控Spark集群
    - http://192.168.199.188:4040/  监控Spark Job

- 代码

  ```python
  from pyspark.sql import SparkSession
  # 255.255.255.255 0~255 256  2^8 8位2进制数
  #将ip转换为特殊的数字形式  223.243.0.0|223.243.191.255|  255 2^8
  #‭11011111‬
  #00000000
  #1101111100000000
  #‭        11110011‬
  #11011111111100110000000000000000
  def ip_transform(ip):     
      ips = ip.split(".")#[223,243,0,0] 32位二进制数
      ip_num = 0
      for i in ips:
          ip_num = int(i) | ip_num << 8
      return ip_num
  
  #二分法查找ip对应的行的索引
  def binary_search(ip_num, broadcast_value):
      start = 0
      end = len(broadcast_value) - 1
      while (start <= end):
          mid = int((start + end) / 2)
          if ip_num >= int(broadcast_value[mid][0]) and ip_num <= int(broadcast_value[mid][1]):
              return mid
          if ip_num < int(broadcast_value[mid][0]):
              end = mid
          if ip_num > int(broadcast_value[mid][1]):
              start = mid
  
  def main():
      spark = SparkSession.builder.appName("test").getOrCreate()
      sc = spark.sparkContext
      city_id_rdd = sc.textFile("file:///home/hadoop/app/tmp/data/ip.txt").map(lambda x:x.split("|")).map(lambda x: (x[2], x[3], x[13], x[14]))
      #创建一个广播变量
      city_broadcast = sc.broadcast(city_id_rdd.collect())
      dest_data = sc.textFile("file:///home/hadoop/app/tmp/data/20090121000132.394251.http.format").map(
          lambda x: x.split("|")[1])
      #根据取出对应的位置信息
      def get_pos(x):
          city_broadcast_value = city_broadcast.value
          #根据单个ip获取对应经纬度信息
          def get_result(ip):
              ip_num = ip_transform(ip)
              index = binary_search(ip_num, city_broadcast_value)
              #((纬度,精度),1)
              return ((city_broadcast_value[index][2], city_broadcast_value[index][3]), 1)
  
          x = map(tuple,[get_result(ip) for ip in x])
          return x
  
      dest_rdd = dest_data.mapPartitions(lambda x: get_pos(x)) #((纬度,精度),1)
      result_rdd = dest_rdd.reduceByKey(lambda a, b: a + b)
      print(result_rdd.collect())
      sc.stop()
  
  if __name__ == '__main__':
      main()
  ```

- **广播变量的使用**

  - 要统计Ip所对应的经纬度, 每一条数据都会去查询ip表
  - 每一个task 都需要这一个ip表, 默认情况下, 所有task都会去复制ip表
  - 实际上 每一个Worker上会有多个task, 数据也是只需要进行查询操作的, 所以这份数据可以共享,没必要每个task复制一份
  - 可以通过广播变量, 通知当前worker上所有的task, 来共享这个数据,避免数据的多次复制,可以大大降低内存的开销
  - sparkContext.broadcast(要共享的数据)
