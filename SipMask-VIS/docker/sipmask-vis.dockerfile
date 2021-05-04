ARG PYTORCH="1.1.0"
ARG CUDA="10.0"
ARG CUDNN="7.5"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6

# Install mmdetection
RUN conda install cython -y



COPY setup.py README.md  /SipMask-VIS/
COPY mmdet/ /SipMask-VIS/mmdet/
WORKDIR /SipMask-VIS

RUN pip install -e .
RUN pip uninstall -y pycocotools

RUN pip install  git+https://github.com/youtubevos/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
RUN pip install  --upgrade mmcv==0.2.12
# remove old opencv without ffmpeg
RUN pip uninstall -y opencv-python
# install opencv version with ffmpeg supportconda 
RUN conda install -c conda-forge opencv -y 
RUN pip  install debugpy
EXPOSE 5678


