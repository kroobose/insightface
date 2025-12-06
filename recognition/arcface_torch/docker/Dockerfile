# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-01.html
FROM nvcr.io/nvidia/pytorch:24.01-py3

#
# settings
#
RUN apt-get update && apt-get -y upgrade && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-dev tzdata fonts-ipafont zstd tree tmux \
    && ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /usr/local/src/*
ENV TZ=Asia/Tokyo

#
# ClearML
#
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir \
     clearml==1.13.1 \
     boto3 \
     hydra-core==1.3.2

# common DL packages
RUN pip3 install --no-cache-dir \
        easydict==1.13 \
        'numpy<2' \
        mxnet==1.6.0 \
        Cython \
        matplotlib \
        scikit-learn \
        PrettyTable \
        ipykernel \
        protobuf \
        optuna

# opencv...
RUN pip3 install --no-cache-dir \
    "opencv-python-headless<4.3" \
    pyyaml \
    scipy==1.8.1 \
    tensorboard \
    termcolor \
    timm==0.4.12 \
    yacs
