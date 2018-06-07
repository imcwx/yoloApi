FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update -y && apt-get upgrade -y && apt-get install -y python-dev python-pip python3-dev python3-pip git libsm6 libxext6 libxrender-dev \
        && pip3 install opencv-python && pip3 install Keras flask requests

RUN mkdir /yoloApi
COPY ./yoloApi /yoloApi
WORKDIR /yoloApi


# RUN apt-get update && apt-get install -y protobuf-compiler python-pil python-lxml python-tk git \
#         && pip install --upgrade pip \
#         && pip install Cython \
#         && pip install jupyter \
#         && pip install matplotlib

# RUN mkdir /tensorflow \
#         && cd /tensorflow \
#         && git clone https://github.com/tensorflow/models.git \
#         && git clone https://github.com/cocodataset/cocoapi.git \
#         && cd cocoapi/PythonAPI \
#         && make \
#         && cp -r pycocotools /tensorflow/models/research/ \
#         && cd /tensorflow/models/research/

# COPY ./app /tensorflow/

# WORKDIR /tensorflow/app/

# ENTRYPOINT [ "python", "anders.py" ]

# RUN protoc object_detection/protos/*.proto --python_out=.
