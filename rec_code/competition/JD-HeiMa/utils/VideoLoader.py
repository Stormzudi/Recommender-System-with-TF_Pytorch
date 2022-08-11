import urllib.request
import urllib.parse
import cv2
import os
import numpy as np
import math
import string
import io
import random
from PIL import Image


class VideoLoader(object):
    def __init__(self, ):
        self.local_path = ".."
        self.folder_name = None


    # 未使用
    def _save_image(self, image_path, image):
        """Save the images.
            Args:
                num: serial number
                image: image resource
            Returns:
                None
            """
        # image_path = '{}/images/{}.jpg'.format(self.local_path, str(num))
        cv2.imwrite(image_path, image)
        print('save_success')
        pass

    # 给视频切桢
    def _frame_capture(self, videos):
        # 1. Record video path
        videos_path = './static/source_videos/'

        # videos = os.listdir(videos_path)

        # 2. Record images path
        # folder_name = os.path.join(self.static_path, 'images')
        # self.images_path = folder_name
        # os.makedirs(folder_name, exist_ok=True)

        result = {}
        for video_name in videos:

            file_name = video_name.split('.')[0]
            # os.makedirs(os.path.join(folder_name, file_name), exist_ok=True)

            # 3. Read video

            vc = cv2.VideoCapture(videos_path + video_name)

            # loop read video frame
            frame_interval = 12  # frame cell
            frame_interval_count = 0
            frames_num = int(vc.get(7) // frame_interval)

            # determine whether to open normally
            if vc.isOpened():
                ret, frame = vc.read()
                # output matraix
                output = np.zeros((frames_num + 1, frame.shape[0], frame.shape[1], 3), dtype="uint8")
            else:
                ret = False

            count = 0
            while ret:
                ret, frame = vc.read()
                # reshape (height, weight)
                # frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)  # reshape
                # store operation every time f frame
                if frame_interval_count % frame_interval == 0 and ret:
                    # image_path = '{}/{}.jpg'.format(folder_name, str(count))
                    # self._save_image(image_path, frame)

                    # add matrix
                    output[count, :] = frame
                    count += 1

                frame_interval_count += 1
            result.update({
                "{}".format(file_name): output
            })
        return result

    # 传入视频名称数组['nn.mp4','jd.mp4']
    # 上传视频立刻执行切桢,返回字典
    def extract(self, videos_name_list):
        """
        :type video_name: list
        """
        # 1. extract all the frames video
        videos = self._frame_capture(videos_name_list)  # shape: [-1, h, w, 3]

        # 字典
        for video, video_embed in videos.items():

            # 2. do for each video
            bs = list()
            for idx, frame_tensor in enumerate(video_embed):
                max_size = 240
                img = Image.fromarray(frame_tensor)
                if img.size[0] > img.size[1]:
                    width = max_size
                    height = math.ceil(max_size / img.size[0] * img.size[1])
                else:
                    height = max_size
                    width = math.ceil(max_size / img.size[1] * img.size[0])
                img = img.resize((width, height))
                ts = np.asarray(img).astype('uint8')
                # print(ts.shape)
                bs.append(ts[np.newaxis, :])

                # save images 未使用方法
                # image_path = '{}/{}.jpg'.format(os.path.join(self.images_path, video), idx)
                # self._save_image(image_path, ts)

            videos[video] = np.concatenate(bs, axis=0)

        return videos

if __name__ == '__main__':
    v = VideoLoader()
    res = v.extract(['jd.mp4'])
    pass
