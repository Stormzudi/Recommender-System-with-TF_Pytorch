### 2.1 HDFS的使用

- 启动HDFS

  - 来到$HADOOP_HOME/sbin目录下
  - 执行start-dfs.sh

  ```shell
  [hadoop@hadoop00 sbin]$ ./start-dfs.sh
  ```

  - 可以看到 namenode和 datanode启动的日志信息

  ```shell
  Starting namenodes on [hadoop00]
  hadoop00: starting namenode, logging to /home/hadoop/app/hadoop-2.6.0-cdh5.7.0/logs/hadoop-hadoop-namenode-hadoop00.out
  localhost: starting datanode, logging to /home/hadoop/app/hadoop-2.6.0-cdh5.7.0/logs/hadoop-hadoop-datanode-hadoop00.out
  Starting secondary namenodes [0.0.0.0]
  0.0.0.0: starting secondarynamenode, logging to /home/hadoop/app/hadoop-2.6.0-cdh5.7.0/logs/hadoop-hadoop-secondarynamenode-hadoop00.out
  ```

  - 通过jps命令查看当前运行的进程

  ```shell
  [hadoop@hadoop00 sbin]$ jps
  4416 DataNode
  4770 Jps
  4631 SecondaryNameNode
  4251 NameNode
  ```

  - 可以看到 NameNode DataNode 以及 SecondaryNameNode 说明启动成功

- 通过可视化界面查看HDFS的运行情况

  - 通过浏览器查看 主机ip:50070端口 

  ![1551174774098](/img/hadoop-state.png)

  - Overview界面查看整体情况

  ![1551174978741](/img/hadoop-state1.png)

  - Datanodes界面查看datanode的情况

    ![1551175081051](/img/hadoop-state2.png)

