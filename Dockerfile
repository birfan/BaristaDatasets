# Start from a core gpu-enabled stack version
# FROM gitlab.ilabt.imec.be:4567/ilabt/gpu-docker-stacks/base-notebook
# Set user as root so next commands run correctly
# prepare system for Ubuntu 16.04 with python3.6
FROM ubuntu:16.04
# Set user as root so next commands run correctly
USER root

# install python3.6 and python3.6-dev
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y apt-utils build-essential python3.6 python3.6-dev python3-pip python3.6-venv cmake make wget git

# update pip
RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install wheel
RUN pip3 install --upgrade setuptools

# install pytorch
RUN pip3 install torch==1.1.0 torchvision==0.3.0 -f https://download.pytorch.org/whl/cu100/torch_stable.html

# install tensorflow 1.13.1
RUN pip3 install tensorflow-gpu==1.13.1

RUN git clone https://github.com/facebookresearch/ParlAI.git /project/ParlAI
RUN cd /project/ParlAI; pip3 install -r ./requirements.txt; echo "" > README.md; python3.6 setup.py develop

COPY . /project/hrinlp/
COPY ./parlai_internal/ /project/ParlAI/parlai_internal
COPY ./data/ /project/ParlAI/data

RUN pip3 install -r /project/hrinlp/requirements.txt

RUN ln -s /usr/local/bin/python3 /usr/bin/python & \
    ln -s /usr/local/bin/pip3 /usr/bin/pip


