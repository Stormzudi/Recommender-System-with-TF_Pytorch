from flask import Flask, session, render_template
from controllers.PositionController import position
from controllers.SourceController import source_videos
import os
from flask_cors import CORS

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
os.environ["NCCL_DEBUG"] = "INFO"

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


app.register_blueprint(position, url_prefix='/position')
app.register_blueprint(source_videos, url_prefix='/source')
# app.register_blueprint(output_videos, url_prefix='/output')



CORS(app, resources={r"/*"})
CORS(source_videos, resources={r"/source/*"})


# 渲染页面
@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('index.html', name=name)


"""
运行方式：
（1）启动下面main函数
（2）在powshell中， 运行其他文件名称的程序：
    $env:FLASK_APP = "study_flask"
    flask run
"""

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=30208)
