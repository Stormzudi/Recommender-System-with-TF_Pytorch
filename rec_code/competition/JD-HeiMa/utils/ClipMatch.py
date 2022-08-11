import torch
import clip
from utils.VideoLoader import VideoLoader
import numpy as np
from torch import Tensor
from PIL import Image
import cv2
import os

"""
@ File : ClipMatch.py
@ Description :     # 调用VideoLoader切割视频，得到图片矩阵，然后计算得分


"""
class CLIPTextEncoder(object):

    def __init__(self,
                 pretrained_model_name_or_path: str = 'ViT-B/32',
                ):
        self.videos_path = None
        self.videoLoader = VideoLoader()
        self.Images = None
        self.static_path = None
        self.probs_list = dict()  # the frame scores
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocessor = clip.load(pretrained_model_name_or_path, device=self.device)
        self.preprocessor = preprocessor
        self.model = model

    def score(self, image_features, text_features):
        """

        :param image_features: shape: (dim, 512)
        :param text_features: shape: (1, 512)
        :return: score
        """
        logit_scale = self.model.logit_scale.exp()
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logits_per_image = logit_scale * image_features @ text_features.t()
        probs = logits_per_image.softmax(dim=0).cpu().detach().numpy()

        # print(" img Label probs:", probs)
        return probs



    def func(self, image_features
             , text_features
             ):
        """
        :param image_features: shape: (dim, 512)
        :param text_features: shape: (1, 512)
        :return: score
        """

        with torch.no_grad():
            # image_features = model.encode_image(image)
            # text_features = model.encode_text(text)

            logits_per_image, logits_per_text = self.model(image_features, text_features)
            probs = logits_per_image.softmax(dim=0).cpu().numpy()
        print("Label probs:", probs)
        return probs

    # 调用VideoLoader切割视频，得到图片矩阵，然后计算得分
    # video_name = "[mm.mp4]"

    def clip_extract(self, token, videos):
        """
        :param token: words of searching videos
        :param videos: [-1, h, w, 3]
        :return:
        """
        # 1. get videos tokens
        tensor_images_features = []
        for image_features in videos:
            image = Image.fromarray(image_features.astype('uint8')).convert('RGB')
            # 2. reshape images
            image_features = self.preprocessor(image).unsqueeze(0).to(self.device)
            # tensor = self.model.encode_image(image_features)
            tensor_images_features.append(image_features)

        tensor_images = torch.cat(tensor_images_features)
        # 3. get text token
        text = clip.tokenize([token]).to(self.device)
        # text_features = self.model.encode_text(text)

        probs = self.func(tensor_images, text)
        return probs

    def split_video(self, id, min_len=11, nums=1):
        """

        :param id:  the video name
        :param min_len: the min's split of the video
        :param nums: return nums of each video
        """
        # 1. get video
        videos_path = os.path.join(self.static_path, 'source_videos')
        vc = cv2.VideoCapture(os.path.join(videos_path, "{}.mp4".format(id)))

        # 2. get saved `split videos`
        s_path = os.path.join(self.static_path, "splits")
        os.makedirs(s_path, exist_ok=True)

        fps = vc.get(cv2.CAP_PROP_FPS)
        width = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
        size = (int(width), int(height))

        # 3. define the split-video pathways
        if nums > 1:
            pass
        else:
            split_path = os.path.join(s_path, id, )
            os.makedirs(split_path, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
            videoWriter = cv2.VideoWriter(split_path+'/'+"{}_{}.mp4".format(id, nums), fourcc, fps, size)

            if vc.isOpened():
                ret, frame = vc.read()
            else:
                ret = False

            # 4. DO !!!
            i = 0
            while ret:
                success, frame = vc.read()
                if success:
                    i += 1
                    # start = 27, end = 37
                    if (i >= 27 * fps and i <= 37 * fps):
                        videoWriter.write(frame)
                else:
                    break

    def clip_rum_model(self, token, videos=["jd.mp4"]):
        """
            run clip model
            target:
                1. get scores of each video with matching the `token`
                2. split the video and save all the split video
        :param token: words of searching videos
        :return:
        """
        # RUN videoLoader.py TO get videos information
        videos = self.videoLoader.extract(videos_name_list=videos)
        self.static_path = self.videoLoader.static_path

        # videos: dict
        for id, video_emb in videos.items():
            self.probs_list.update({
                id:self.clip_extract(token, video_emb)
            })

            # SPLIT VIDEOS
            self.split_video(id, min_len=11, nums=1)

if __name__ == '__main__':
    c = CLIPTextEncoder()
    res = c.clip_rum_model(token="a men with a computer")
    print(res)