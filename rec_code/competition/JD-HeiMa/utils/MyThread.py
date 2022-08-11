# @Author  : sichaolong
# @File    : MyThread.py
# @Software: PyCharm
import threading

# 自定义线程类，方便获取结果
class MyThread(threading.Thread):
    def __init__(self,func,args=()):
        super(MyThread,self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.result = self.func(*self.args)
    def get_result(self):
        try:
            return self.result
        except Exception:
            return None