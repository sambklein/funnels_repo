ARG TORCH_VERSION=1.8.1-cuda11.1-cudnn8-runtime

FROM pytorch/pytorch:${TORCH_VERSION}#this will get replaced in the ci though

USER root
RUN apt-get -qq -y update && \
    apt-get -qq -y upgrade && \
    apt-get -qq -y install \
        wget \
        curl \
        git \
        sudo \
        unzip \
        p7zip-full \
        fastjar

COPY docker/requirements.txt .
COPY . .

RUN python3 -m pip install -r requirements.txt

#RUN mkdir python_install
#ENV PYTHONPATH=/workspace:/workspace/python_install:${PYTHONPATH}
#RUN python3 setup.py develop --install-dir python_install

EXPOSE 8888
