import torch
import clip
from videoLoader.videoLoader import VideoLoader
from multilingual_clip import pt_multilingual_clip
import numpy as np
from torch import Tensor
from PIL import Image
import cv2
import os


class CLIPTextEncoder(object):
    def __init__(self,
                 pretrained_model_name_or_path: str = 'ViT-B/32',
                 ):
        self.videoLoader = VideoLoader()
        self.token = None
        self.Images = None
        self.probs = None   # the frame scores
        self.video_caputre = None
        self.video_dir = None

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

            logits_per_image, logits_per_text = self.model(image_features, text_features)
            probs = logits_per_image.softmax(dim=0).cpu().numpy()
        self.probs = probs
        print("Label probs:", probs)
        return probs

    def clip_extract(self, video_name, token="a man with a computer"):
        self.token = token

        # 1. run videoLoader.py TO get video information
        video_mat = self.videoLoader.extract(video_name)
        self.video_caputre = self.videoLoader.video_caputre
        self.video_dir = self.videoLoader.videos_dir

        tensor_images_features = []
        # 2. get videos tokens
        for image_features in video_mat:
            image = Image.fromarray(image_features.astype('uint8')).convert('RGB')
            # 2. reshape images
            image_features = self.preprocessor(image).unsqueeze(0).to(self.device)
            # tensor = self.encode.encode_image(image_features)
            tensor_images_features.append(image_features)

        tensor_images = torch.cat(tensor_images_features)
        # 3. get text token
        text = clip.tokenize([self.token]).to(self.device)
        # text_features = self.encode.encode_text(text)

        probs = self.func(tensor_images, text)
        return probs

    def split_video(self,):
        # get video parameters
        fps = self.video_caputre.get(5)
        width = self.video_caputre.get(3)
        height = self.video_caputre.get(4)
        size = (int(width), int(height))

        # the split-video pathways
        os.makedirs(self.video_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        videoWriter = cv2.VideoWriter(self.video_dir, fourcc, fps, size)

        i = 0
        while True:
            success, frame = self.video_caputre.read()
            if success:
                i += 1
                if (i >= 27 * fps and i <= 37 * fps):
                    videoWriter.write(frame)
            else:
                break


if __name__ == '__main__':
    c = CLIPTextEncoder()
    res = c.clip_extract(video_name="mm.mp4")
    print(res)

    c.split_video()
