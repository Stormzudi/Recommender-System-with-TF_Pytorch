## MapReduce实战

### 3.3.1 利用MRJob编写和运行MapReduce代码

**mrjob 简介**

- 使用python开发在Hadoop上运行的程序, mrjob是最简单的方式
- mrjob程序可以在本地测试运行也可以部署到Hadoop集群上运行
- 如果不想成为hadoop专家, 但是需要利用Hadoop写MapReduce代码,mrJob是很好的选择

**mrjob 安装**

- 使用pip安装
  - pip install mrjob

**mrjob实现WordCount**

```
from mrjob.job import MRJob

class MRWordFrequencyCount(MRJob):

    def mapper(self, _, line):
        yield "chars", len(line)
        yield "words", len(line.split())
        yield "lines", 1

    def reducer(self, key, values):
        yield key, sum(values)


if __name__ == '__main__':
    MRWordFrequencyCount.run()
```

**运行WordCount代码**

打开命令行, 找到一篇文本文档, 敲如下命令:

```shell
python mr_word_count.py my_file.txt
```

### 3.3.2 运行MRJOB的不同方式

1、内嵌(-r inline)方式

特点是调试方便，启动单一进程模拟任务执行状态和结果，默认(-r inline)可以省略，输出文件使用 > output-file 或-o output-file，比如下面两种运行方式是等价的

python word_count.py -r inline input.txt > output.txt
python word_count.py input.txt > output.txt

2、本地(-r local)方式

用于本地模拟Hadoop调试，与内嵌(inline)方式的区别是启动了多进程执行每一个任务。如：

python word_count.py -r local input.txt > output1.txt

3、Hadoop(-r hadoop)方式

用于hadoop环境，支持Hadoop运行调度控制参数，如：

1)指定Hadoop任务调度优先级(VERY_HIGH|HIGH),如：--jobconf mapreduce.job.priority=VERY_HIGH。

2)Map及Reduce任务个数限制，如：--jobconf mapreduce.map.tasks=2  --jobconf mapreduce.reduce.tasks=5

python word_count.py -r hadoop hdfs:///test.txt -o  hdfs:///output

### 3.3.3 mrjob 实现 topN统计（实验）

统计数据中出现次数最多的前n个数据

```python
import sys
from mrjob.job import MRJob,MRStep
import heapq

class TopNWords(MRJob):
    def mapper(self, _, line):
        if line.strip() != "":
            for word in line.strip().split():
                yield word,1

    #介于mapper和reducer之间，用于临时的将mapper输出的数据进行统计
    def combiner(self, word, counts):
        yield word,sum(counts)

    def reducer_sum(self, word, counts):
        yield None,(sum(counts),word)

    #利用heapq将数据进行排序，将最大的2个取出
    def top_n_reducer(self,_,word_cnts):
        for cnt,word in heapq.nlargest(2,word_cnts):
            yield word,cnt
    
	#实现steps方法用于指定自定义的mapper，comnbiner和reducer方法
    def steps(self):
        return [
            MRStep(mapper=self.mapper,
                   combiner=self.combiner,
                   reducer=self.reducer_sum),
            MRStep(reducer=self.top_n_reducer)
        ]

def main():
    TopNWords.run()

if __name__=='__main__':
    main()
```









