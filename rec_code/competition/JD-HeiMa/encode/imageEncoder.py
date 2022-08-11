import torch
import numpy as np
from PIL import Image
import clip


class CLIPImageEncoder(object):
    def __init__(self,
                 pretrained_model_name_or_path: str = 'ViT-L/14',
                 ):
        self.Images = None
        self.probs = None  # the frame scores
        self.video_dir = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocessor = clip.load(pretrained_model_name_or_path,
                                        device=self.device)
        # model, _, preprocessor = open_clip.create_model_and_transforms(pretrained_model_name_or_path, pretrained="laion400m_e32", device=torch.device(self.device))
        self.preprocessor = preprocessor
        self.model = model
        self.logit_scale = self.model.logit_scale.exp()

    def clip_extract(self, pixel_matrices):
        tensor_images_features = []
        # get videos tokens
        for pixel_matrix in pixel_matrices:
            image = self.preprocessor(Image.fromarray(pixel_matrix)).unsqueeze(
                0).to(self.device)
            image_features = self.model.encode_image(image)
            tensor_images_features.append(image_features.cpu().detach())
        tensor_images_features = np.concatenate(tensor_images_features, axis=0)
        return tensor_images_features
