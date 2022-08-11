from flask import jsonify
from config import *
from GetIndex import *
from videoLoader import videoLoader
from encode import textEncoder
from encode import imageEncoder


videoloader = videoLoader.VideoLoader()
image_encode = imageEncoder.CLIPImageEncoder()
text_encode = textEncoder.CLIPTextEncoder()
logit_scale = image_encode.logit_scale.cpu()

class PositionVideoService(object):
    def __init__(self, ):
        pass

    def resp_position(requ_data):
        # 结果
        videos_array = {}
        file_name_list = requ_data['file_name_list']
        text = requ_data['key_word']
        # TODO 调用clip模型，返回视频片段
        # videos_array = ["./static/output_videos/jd/sreamlit_dmeo1.mp4","./static/output_videos/jd/sreamlit_dmeo2.mp4","./static/output_videos/jd/sreamlit_dmeo3.mp4"]
        # 关键字
        text_list = [text]
        text_features = text_encode.clip_extract(text_list)
        # 模型比对
        for i, text_feature in enumerate(text_features):
            for j, video_name in enumerate(file_name_list):

                image_features = video_features[video_name]
                probs = score(logit_scale, image_features, text_feature)
                index_list = getMultiRange(probs)
                videoloader.split_video(video_name.split('.')[0], index_list)
        #  在 root_path/static/splits/ 文件夹下 新建文件夹如 demo,然后将demo.mp4的定位视频片段保存下来
        for file_name in file_name_list:
            dir_path = splits_cache_path.join('/{}'.format(file_name.split('.')[0]))
            file_list = os.listdir(os.path.join(dir_path))
            videos_array[file_name] = [os.path.join(dir_path,file) for file in file_list]
        a = 1
        return jsonify({'code': '200', 'msg': '定位成功~', 'data': videos_array})




