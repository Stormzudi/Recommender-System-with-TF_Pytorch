### 2.2 HDFS shell操作

- 调用文件系统(FS)Shell命令应使用 bin/hadoop fs <args>的形式

  - ### ls

    使用方法：hadoop fs -ls <args>

    如果是文件，则按照如下格式返回文件信息：
    文件名 <副本数> 文件大小 修改日期 修改时间 权限 用户ID 组ID 
    如果是目录，则返回它直接子文件的一个列表，就像在Unix中一样。目录返回列表的信息如下：
    目录名 <dir> 修改日期 修改时间 权限 用户ID 组ID 
    示例：
    hadoop fs -ls /user/hadoop/file1 /user/hadoop/file2 hdfs://host:port/user/hadoop/dir1 /nonexistentfile 
    返回值：
    成功返回0，失败返回-1。 

  - ### text

    使用方法：hadoop fs -text <src> 

    将源文件输出为文本格式。允许的格式是zip和TextRecordInputStream。

  - ### mv

    使用方法：hadoop fs -mv URI [URI …] <dest>

    将文件从源路径移动到目标路径。这个命令允许有多个源路径，此时目标路径必须是一个目录。不允许在不同的文件系统间移动文件。 
    示例：

    - hadoop fs -mv /user/hadoop/file1 /user/hadoop/file2
    - hadoop fs -mv hdfs://host:port/file1 hdfs://host:port/file2 hdfs://host:port/file3 hdfs://host:port/dir1

    返回值：

    成功返回0，失败返回-1。

  - ### put

    使用方法：hadoop fs -put <localsrc> ... <dst>

    从本地文件系统中复制单个或多个源路径到目标文件系统。也支持从标准输入中读取输入写入目标文件系统。

    - hadoop fs -put localfile /user/hadoop/hadoopfile
    - hadoop fs -put localfile1 localfile2 /user/hadoop/hadoopdir
    - hadoop fs -put localfile hdfs://host:port/hadoop/hadoopfile
    - hadoop fs -put - hdfs://host:port/hadoop/hadoopfile 
      从标准输入中读取输入。

    返回值：

    成功返回0，失败返回-1。

  - ### rm

    使用方法：hadoop fs -rm URI [URI …]

    删除指定的文件。只删除非空目录和文件。请参考rmr命令了解递归删除。
    示例：

    - hadoop fs -rm hdfs://host:port/file /user/hadoop/emptydir

    返回值：

    成功返回0，失败返回-1。

- http://hadoop.apache.org/docs/r1.0.4/cn/hdfs_shell.html

### 2.4.1 HDFS shell操作练习

- 在centos 中创建 test.txt  

  ```shell
  touch test.txt
  ```

- 在centos中为test.txt 添加文本内容

  ```shell
  vi test.txt
  ```

- 在HDFS中创建 hadoop001/test 文件夹

  ``` shell
  hadoop fs -mkdir -p /hadoop001/test
  ```

- 把text.txt文件上传到HDFS中

  ```shell
  hadoop fs -put test.txt /hadoop001/test/
  ```

- 查看hdfs中 hadoop001/test/test.txt 文件内容

  ```shell
  hadoop fs -cat /hadoop001/test/test.txt
  ```

- 将hdfs中 hadoop001/test/test.txt文件下载到centos

   ```shell
   hadoop fs -get /hadoop001/test/test.txt test.txt
   ```

- 删除HDFS中 hadoop001/test/

   hadoop fs -rm -r /hadoop001

