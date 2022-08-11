import urllib.parse
import urllib.request
import io


url = 'https://www.jitabang.com/jiaoxue/17736.html'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
tmp_fn = "nn.mp4"
with urllib.request.urlopen(req) as fp:
    buffer = fp.read()
    a = io.BytesIO(buffer)

    with open(tmp_fn, 'wb') as f:
        f.write(a.read())
    b = 1