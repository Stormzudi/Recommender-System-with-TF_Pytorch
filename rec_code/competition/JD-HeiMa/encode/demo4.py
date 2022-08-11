import cv2


"""
ref: 
1. https://blog.csdn.net/weixin_42265958/article/details/115531210
2. https://blog.csdn.net/u010420283/article/details/89706794
"""

def split_video(input_video, output_video):
    video_caputre = cv2.VideoCapture(input_video)

    # fps = 30  # 保存视频的帧率

    # get video parameters
    fps = video_caputre.get(cv2.CAP_PROP_FPS)
    width = video_caputre.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video_caputre.get(cv2.CAP_PROP_FRAME_HEIGHT)

    size = (int(width), int(height))


    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    videoWriter = cv2.VideoWriter(output_video, fourcc, fps, size)
    i = 0

    while True:
        success, frame = video_caputre.read()
        if success:
            i += 1
            if (i >= 300 and i <= 450):  # 截取300帧到450帧的视频
                videoWriter.write(frame)
        else:
            break


if __name__ == '__main__':
    input_file = '../static/videos/mm.mp4'
    output_file = '../static/videos/mm_new.avi'
    split_video(input_video=input_file, output_video=output_file)
