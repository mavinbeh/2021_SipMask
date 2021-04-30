ARG PYTORCH="1.1.0"
ARG CUDA="10.0"
ARG CUDNN="7.5"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6

# Install mmdetection
RUN conda install cython -y
COPY . /mmdetection
WORKDIR /mmdetection/SipMask-VIS
RUN pip install -e .
RUN echo "x"
RUN pip uninstall -y pycocotools
RUN pip install  git+https://github.com/youtubevos/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
RUN pip install  --upgrade mmcv==0.2.12
