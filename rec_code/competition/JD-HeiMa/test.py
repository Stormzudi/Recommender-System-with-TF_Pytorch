# 上传端
# 1. 从前端导入视频
# 2. 对视频进行本地化保存
# 3. 得到每个视频的帧的array
# 4. 把array传给 clip的image_encode,得到embedding
# 5. 把embedding 传给faiss保存

# 搜索端
# 1.从前端导入text
# 2.把text进行embedding
# 3.对每个faiss 文件进行搜索，得到每个文件高于阈值的相似度的帧的索引
# 3.返回索引对应时间戳
# 5.对每个视频切帧
# 5.返回给前端
import os.path

import torch

from config import *
from GetIndex import *
from videoLoader import videoLoader
from encode import textEncoder
from encode import imageEncoder

import faiss

videoloader = videoLoader.VideoLoader()
image_encode = imageEncoder.CLIPImageEncoder()
text_encode = textEncoder.CLIPTextEncoder()
logit_scale = image_encode.logit_scale.cpu()


# 存储video_features
video_name_list = ["ME1658308190408"]
def video_processor(video_name_list):
    video_name_list = ["ME1658308190408.mp4"]
    for video_name in video_name_list:
        res = videoloader.extract(video_name)
        image_features = image_encode.clip_extract(res)
        video_features[video_name] = image_features


video_processor("name")

text_list = ["雕像"]
text_features = text_encode.clip_extract(text_list)
for i, text_feature in enumerate(text_features):
    for j, video_name in enumerate(video_features.keys()):

        image_features = video_features[video_name]
        probs = score(logit_scale,image_features, text_feature)
        index_list = getMultiRange(probs)
        videoloader.split_video(video_name.split('.')[0],index_list)
        a = 1
for file_name in video_name_list:
    path = os.path.join(root_path,'static/splits/{}'.format(file_name))
    k = os.listdir(path)
    a= 1
print(text_features.shape)
