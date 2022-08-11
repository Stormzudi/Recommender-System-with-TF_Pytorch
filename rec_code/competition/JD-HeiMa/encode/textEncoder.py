import torch
from videoLoader.videoLoader import VideoLoader
from multilingual_clip import pt_multilingual_clip
import transformers
import numpy as np
import os


class CLIPTextEncoder(object):
    def __init__(self,
                 pretrained_model_name_or_path: str = "M-CLIP/XLM-Roberta-Large-Vit-L-14",
                 ):
        self.Images = None
        self.probs = None   # the frame scores
        self.video_dir = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(pretrained_model_name_or_path)
        tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = tokenizer
        self.model = model


    def clip_extract(self, text_list):

        tensor_text_features = []
        # 2. get videos tokens
        for text in text_list:
            text_feature = self.model.forward(text, self.tokenizer)
            tensor_text_features.append(text_feature.cpu().detach())
        tensor_text_features = np.concatenate(tensor_text_features, axis=0)
        return tensor_text_features

if __name__ == "__main__":
    text_list = ["我爱中国","京东集团","新南威尔士大学"]
    text_encode = CLIPTextEncoder()
    text_features = text_encode.clip_extract(text_list)
    print(text_features.shape)


