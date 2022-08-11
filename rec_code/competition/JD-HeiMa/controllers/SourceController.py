
from flask import jsonify, request, app, Blueprint
from services.SourceVideoService import SourceVideoService



source_videos = Blueprint('source',__name__)
s = SourceVideoService()

@source_videos.route('/upload', methods=['POST'])
def source_video_upload():
    requ_data = {
        'files': request.files
    }
    resp_data = s.resp_upload(requ_data)
    return resp_data

"""
@ Author : sichaolong
@ File : SourceController.py
@ Description : 获取视频名字列表 

{
    "code": "200",
    "data": [
        "clip前端演示demo.mp4"
    ],
    "msg": "成功~"
}
"""

@source_videos.route('/names', methods=['GET'])
def source_videos_name_list():
    resp_data = s.resp_name_list()
    return resp_data

"""
@ Author : sichaolong
@ File : SourceController.py
@ Description : 删除视频

请求格式：{"file_name": "clip前端演示demo.mp4"}

"""
@source_videos.route('/delete', methods=['POST'])
def source_video_delete():
    requ_data = {
        'file_name': request.form['file_name']
    }
    resp_data = s.resp_delete(requ_data)
    return resp_data


"""
@ Author : sichaolong
@ File : SourceController.py
@ Description : 前端点击视频名字 获取 视频磁盘绝对路径 xxx/xxx/static/videos/xxx.mp4

"""
@source_videos.route('/preview', methods=['POST'])
def source_video_preview():
    requ_data = {
        'file_name': request.form['file_name']
    }
    resp_data = s.resp_preview(requ_data)
    return resp_data


