### 2.5 HDFS环境搭建

- 下载jdk 和 hadoop 放到 ~/software目录下 然后解压到 ~/app目录下

  ```shell
  tar -zxvf 压缩包名字 -C ~/app/
  ```

- 配置环境变量

  ```shell
  vi ~/.bash_profile
  export JAVA_HOME=/home/hadoop/app/jdk1.8.0_91
  export PATH=$JAVA_HOME/bin:$PATH
  export HADOOP_HOME=/home/hadoop/app/hadoop......
  export PATH=$HADOOP_HOME/bin:$PATH
  
  #保存退出后
  source ~/.bash_profile
  ```

- 进入到解压后的hadoop目录 修改配置文件

  - 配置文件作用
    - core-site.xml  指定hdfs的访问方式
    - hdfs-site.xml  指定namenode 和 datanode 的数据存储位置
    - mapred-site.xml 配置mapreduce
    - yarn-site.xml  配置yarn

  - 修改hadoop-env.sh

  ```shell
  cd etc/hadoop
  vi hadoop-env.sh
  #找到下面内容添加java home
  export_JAVA_HOME=/home/hadoop/app/jdk1.8.0_91
  ```

  - 修改 core-site.xml 在 <configuration>节点中添加

  ```xml
  <property>
    <name>fs.default.name</name>
    <value>hdfs://hadoop000:8020</value>
  </property>
  ```

  - 修改hdfs-site.xml 在 configuration节点中添加

  ```xml
  <property>
      <name>dfs.namenode.name.dir</name>
      <value>/home/hadoop/app/tmp/dfs/name</value>
  </property>
  <property>
      <name>dfs.datanode.data.dir</name>
      <value>/home/hadoop/app/tmp/dfs/data</value>
  </property>
  <property>
      <name>dfs.replication</name>
      <value>1</value>
  </property>
  ```

  - 修改 mapred-site.xml 
  - 默认没有这个 从模板文件复制 

  ```shell
  cp mapred-site.xml.template mapred-site.xml
  ```

  ​	在mapred-site.xml  的configuration 节点中添加

  ```xml
  <property>
      <name>mapreduce.framework.name</name>
      <value>yarn</value>
  </property>
  ```

  - 修改yarn-site.xml configuration 节点中添加

  ```xml
  <property>
      <name>yarn.nodemanager.aux-services</name>
      <value>mapreduce_shuffle</value>
  </property>
  ```

- 来到hadoop的bin目录

  ```shell
  ./hadoop namenode -format (这个命令只运行一次)
  ```

- 启动hdfs 进入到  sbin

  ```shell
  ./start-dfs.sh
  ```

- 启动启动yarn 在sbin中

