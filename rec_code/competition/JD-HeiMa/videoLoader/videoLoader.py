import time
import cv2
import os
import numpy as np
import math
import sys
from PIL import Image

sys.path.append("..")
from config import *


class VideoLoader(object):
    def __init__(self):
        self.local_path = root_path
        self.folder_name = None
        self.video_caputre = None
        self.videos_dir = None
        self.static_path = static_cache_path
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

    def _save_images_matraix(self, ):

        pass

    def _frame_capture(self, videos):
        videos_path = "{}/static/videos/".format(self.local_path)
        # videos = os.listdir(videos_path)

        for video_name in videos:

            file_name = video_name# .split('.')[0]
            # videos_path
            self.videos_dir =os.path.join(videos_path , file_name)

            # images_path
            folder_name = "{}/static/images/".format(self.local_path) + file_name
            os.makedirs(folder_name, exist_ok=True)
            self.folder_name = folder_name

            vc = cv2.VideoCapture( self.videos_dir)
            self.video_caputre = vc

            # loop read video frame
            frame_interval = vc.get(cv2.CAP_PROP_FPS)  # frame cell
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
                    image_path = '{}/{}.jpg'.format(folder_name, str(count))
                    # self._save_image(image_path, frame)

                    # add matriex
                    output[count, :] = frame
                    count += 1

                frame_interval_count += 1

            return output

    def extract(self, video_name):
        # extract all the frames video
        t1 = time.time()

        frame_tensors = self._frame_capture([video_name])  # shape: [-1, h, w, 3]

        t2 = time.time()


        bs = list()
        for idx, frame_tensor in enumerate(frame_tensors):

            max_size = resized_size
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

            # save images
            # image_path = '{}/{}.jpg'.format(self.folder_name, idx)
            # self._save_image(image_path, ts)

        frame_tensors = np.concatenate(bs, axis=0)
        return frame_tensors

    def split_video(self, id, video_splits):
        """

        :param id:  the video name
        :param min_len: the min's split of the video
        :param video_splits: list, frames-nums of each video
        """
        # 1. get video
        videos_path = os.path.join(self.static_path, 'videos')
        vc = cv2.VideoCapture(os.path.join(videos_path, "{}.mp4".format(id)))

        # 2. get saved `split videos`
        s_path = os.path.join(self.static_path, "splits")
        os.makedirs(s_path, exist_ok=True)

        fps = vc.get(cv2.CAP_PROP_FPS)
        width = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
        size = (int(width), int(height))

        # 3. define the split-video pathways
        if not video_splits:
            raise "the ``video_splits`` list data is None, Please split "

        # 4. mkdir split_path
        split_path = os.path.join(s_path, id, )
        os.makedirs(split_path, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')

        videoWriters = [cv2.VideoWriter(split_path + '/' + "{}_{}s-{}s.mp4".format(id, left ,right), fourcc, fps, size)
                        for left,right in video_splits]

        # 5. DO !!!
        for vid, vframe in enumerate(video_splits):
            vc = cv2.VideoCapture(os.path.join(videos_path, "{}.mp4".format(id)))
            if vc.isOpened():
                ret, frame = vc.read()
            else:
                ret = False
            i = 0
            while ret:
                success, frame = vc.read()
                if success:
                    i += 1
                    # start = 27, end = 37
                    if (i >= vframe[0] * fps and i <= vframe[-1] * fps):
                        videoWriters[vid].write(frame)
                else:
                    break

if __name__ == '__main__':
    v = VideoLoader()
    res = v.extract("ME1658308190408.mp4")
