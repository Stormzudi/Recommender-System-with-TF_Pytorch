### 3.4 MRJOB 文件合并

**需求描述**

- 两个文件合并 类似于数据库中的两张表合并

```shell
uid uname
01 user1 
02 user2
03 user3
uid orderid order_price
01   01     80
01   02     90
02   03    82
02   04    95
```



**mrjob 实现**

实现对两个数据表进行join操作，显示效果为每个用户的所有订单信息

```
"01:user1"	"01:80,02:90"
"02:user2"	"03:82,04:95"
```

```python
from mrjob.job import MRJob
import os
import sys
class UserOrderJoin(MRJob):
    SORT_VALUES = True
    # 二次排序参数：http://mrjob.readthedocs.io/en/latest/job.html
    def mapper(self, _, line):
        fields = line.strip().split('\t')
        if len(fields) == 2:
            # user data
            source = 'A'
            user_id = fields[0]
            user_name = fields[1]
            yield  user_id,[source,user_name] # 01 [A,user1]
        elif len(fields) == 3:
            # order data
            source ='B'
            user_id = fields[0]
            order_id = fields[1]
            price = fields[2]
            yield user_id,[source,order_id,price] #01 ['B',01,80]['B',02,90]
        else :
            pass

    def reducer(self,user_id,values):
        '''
        每个用户的订单列表
        "01:user1"	"01:80,02:90"
        "02:user2"	"03:82,04:95"

        :param user_id:
        :param values:[A,user1]  ['B',01,80]
        :return:
        '''
        values = [v for v in values]
        if len(values)>1 :
            user_name = values[0][1]
            order_info = [':'.join([v[1],v[2]]) for v in values[1:]] #[01:80,02:90]
            yield ':'.join([user_id,user_name]),','.join(order_info)




def main():
    UserOrderJoin.run()

if __name__ == '__main__':
    main()
```

实现对两个数据表进行join操作，显示效果为每个用户所下订单的订单总量和累计消费金额

```
"01:user1"	[2, 170]
"02:user2"	[2, 177]
```

```python
from mrjob.job import MRJob
import os
import sys
class UserOrderJoin(MRJob):
    # 二次排序参数：http://mrjob.readthedocs.io/en/latest/job.html
    SORT_VALUES = True

    def mapper(self, _, line):
        fields = line.strip().split('\t')
        if len(fields) == 2:
            # user data
            source = 'A'
            user_id = fields[0]
            user_name = fields[1]
            yield  user_id,[source,user_name]
        elif len(fields) == 3:
            # order data
            source ='B'
            user_id = fields[0]
            order_id = fields[1]
            price = fields[2]
            yield user_id,[source,order_id,price]
        else :
            pass



    def reducer(self,user_id,values):
        '''
        统计每个用户的订单数量和累计消费金额
        :param user_id:
        :param values:
        :return:
        '''
        values = [v for v in values]
        user_name = None
        order_cnt = 0
        order_sum = 0
        if len(values)>1:
            for v in values:
                if len(v) ==  2 :
                    user_name = v[1]
                elif len(v) == 3:
                    order_cnt += 1
                    order_sum += int(v[2])
            yield ":".join([user_id,user_name]),(order_cnt,order_sum)



def main():
    UserOrderJoin().run()

if __name__ == '__main__':
    main()	
```