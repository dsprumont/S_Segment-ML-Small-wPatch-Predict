# FROM python:3.7-stretch
FROM continuumio/anaconda3

# install deep learning/common dependencies
RUN conda create -n venv python=3.7
RUN conda activate venv
RUN conda install numpy==1.17.4
RUN conda install -c conda-forge pydicom==1.4.2
RUN conda install pillow==6.2.1
RUN conda install -c conda-forge shapely==1.7.0
RUN conda install pytorch==1.3.1 torchvision==0.4.2 cudatoolkit=10.1 -c pytorch
RUN conda install future==1.17.1
RUN conda install matplotlib==3.1.1

# Install data/models dependencies
RUN mkdir -p /models/resnet50b_fpn256
ADD local/resnet50b_fpn256.json /models/resnet50b_fpn256/config.json
ADD local/Output_Segm_Patch_Resnet50b_fpn256/resnet50b_fpn_final.pth /models/resnet50b_fpn256/weights.pth

RUN chmod 444 /models/resnet50b_fpn256/config.json
RUN chmod 444 /models/resnet50b_fpn256/weights.pth

# Install Cytomine python client
RUN git clone https://github.com/cytomine-uliege/Cytomine-python-client.git && \
    cd /Cytomine-python-client && git checkout tags/v2.5.1 && pip install . && \
    rm -r /Cytomine-python-client

# Install scripts
RUN mkdir -p /app
ADD descriptor.json /app/descriptor.json
ADD utils.py /app/utils.py
ADD run.py /app/run.py
RUN mkdir -p /app/models
ADD models/resnet.py /app/models/resnet.py
ADD models/fpn.py /app/models/fpn.py
ADD models/utils.py /app/models/utils.py
RUN mkdir -p /app/data/detection
ADD data/detection/dataset.py /app/data/detection/dataset.py
ADD data/detection/patch_dataset.py /app/data/detection/patch_dataset.py

ENTRYPOINT ["python3", "/app/run.py"]