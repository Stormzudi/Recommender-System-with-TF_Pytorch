from flask import app, request, jsonify, Blueprint
from services.PositionVideoService import PositionVideoService


"""
视频定位
"""
position = Blueprint('position',__name__)

s = PositionVideoService()
def func(data):
    d = data.replace('[',''). \
        replace(']','')

    a = d.split(',')
    return a


@position.route('/videos', methods=['POST'])
def videos_position():
    requ_data = {
        'file_name_list': func(request.form['file_name_list']),
        'key_word': request.form['key_word']
    }

    resp_data = s.resp_position(requ_data)
    return resp_data