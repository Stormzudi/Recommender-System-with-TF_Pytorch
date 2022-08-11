import os
import threading

from flask import jsonify, session

from utils.MyThread import MyThread
# from utils.VideoLoader import VideoLoader
# sys.path.append("..")
from videoLoader.videoLoader import VideoLoader
from encode.imageEncoder import CLIPImageEncoder
from config import *


"""
上传视频、获取名称集合、获取前端能访问的视频的绝对路径  等逻辑
"""

class SourceVideoService(object):
    def __init__(self, ):
        self.v = VideoLoader()
        self.image_encode = CLIPImageEncoder()
        pass

    # 上传视频到磁盘
    def resp_upload(self, requ_data):
        files = requ_data['files']
        # 是一个字典，file_name为key
        # print(files)
        res = []
        for file_name,file in files.items():
            file_path = video_cache_path + file_name
            if os.path.exists(file_path):
                res.append(file_name + ' 该文件已存在~')
                continue
            try:
                file.save(file_path)
                # 切帧,保存到全局session
                result = self.v.extract(file_name)
                # 提取视频中图像规格，便于之后正确生成视频
                image_features = self.image_encode.clip_extract(result)
                video_features[file_name] = image_features
                # 将上传结果添加到res集合，返回给前端
                res.append(file_name+'该文件已上传~')
            except Exception as e:
                return jsonify({'code': '500', 'msg': '失败~', 'data': e.__cause__})
        return jsonify({'code': '200', 'msg': '成功~', 'data': res})

    # 获取以上传视频名称
    def resp_name_list(self):
        video_name_list = os.listdir(video_cache_path)
        return jsonify({'code': '200', 'msg': '成功~', 'data': video_name_list})

    # 根据视频名称如 'xxx.mp4' 删除视频
    def resp_delete(self, requ_data):
        video_name = requ_data['file_name']
        try:
            os.remove(video_cache_path + video_name)
            del video_features[video_name]
            return jsonify({'code': '200', 'msg': video_name +  '删除成功~', 'data': ''})
        except Exception as e:
            return jsonify({'code': '500', 'msg': video_name + '删除失败~', 'data': ''})

    # 根据视频名称如'xxx.mp4' 得到前端能访问的视频路径 xx/xxx/static/videos/xxx.mp4
    def resp_preview(self, requ_data):
        video_name = requ_data['file_name']
        video_path = video_cache_path + video_name
        if os.path.exists(video_path):
            return jsonify({'code': '200', 'msg': '成功~', 'data': video_path})
        return jsonify({'code': '500', 'msg': '文件不存在~', 'data': ''})






