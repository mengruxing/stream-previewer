# OpenCV Stream Previewer
OpenCV 摄像头预览

## build
with conda
```shell
conda create -n stream-previewer python=3.6
source activate stream-previewer
pip install -r requirements.txt
```

with built in python
```shell
pip install --user -r requirements.txt
```

## setting

edit`cameras.txt`, add the camera src, here's same example:

```text
rtsp://10.0.0.13/stream1
rtsp://10.0.0.14/stream1
0
```

for stream camera
```text
rtsp://
rtmp://
```

for local camera
```text
0
```


## run

```shell
python main.py
```
