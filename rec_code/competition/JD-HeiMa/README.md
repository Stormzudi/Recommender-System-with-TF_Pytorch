#### flask相关说明


##### 1、request

```angular2html
# request.args  get请求提及的数据
# request.form   post请求提交的数据
# request.values  post和get提交的数据总和
# request.cookies  客户端所带的cookie
# request.headers  请求头
# request.path     不带域名，请求路径
# request.full_path  不带域名，带参数的请求路径
# request.script_root  
# request.url           带域名带参数的请求路径
# request.base_url		带域名请求路径
# request.url_root      域名
# request.host_url		域名
# request.host			127.0.0.1:500
# request.files
# obj = request.files['the_file_name']
# obj.save('/var/www/uploads/' + secure_filename(f.filename))
```


##### 2、response

from flask import render_template
from flask import redirect
from flask import make_response
```angular2html
# 响应相关信息
# return "字符串"
# return render_template('html模板路径',**{})
# return redirect('/index.html')
# return jsonify({'k1':'v1'})

# response = make_response(render_template('index.html'))
# response是flask.wrappers.Response类型
# response.delete_cookie('key')
# response.set_cookie('key', 'value')
# response.headers['X-Something'] = 'A value'
# return response
return "内容"
```

