# ---------------------------------------------------#
# 1. Set the base image
# ---------------------------------------------------#
FROM python:3.7-stretch

# ---------------------------------------------------#
# 3. Use pip to install DL/common dependencies
# ---------------------------------------------------#

RUN pip install numpy==1.17.4
RUN pip install pydicom==1.4.2
RUN pip install pillow==6.2.1
RUN pip install shapely==1.7.0
RUN pip install future==0.17.1
RUN pip install matplotlib==3.1.1
RUN pip install torch==1.3.1 torchvision==0.4.2 -f https://download.pytorch.org/whl/torch_stable.html

# ---------------------------------------------------#
# 4. Install Cytomine python client
# ---------------------------------------------------#
RUN git clone https://github.com/Cytomine-ULiege/Cytomine-python-client.git && \
    cd /Cytomine-python-client && git checkout tags/v2.7.3 && pip install . && \
    rm -r /Cytomine-python-client


# ---------------------------------------------------#
# 5. Install data/models dependencies
# ---------------------------------------------------#
RUN mkdir -p /models/resnet50b_fpn256
ADD local/resnet50b_fpn256.json /models/resnet50b_fpn256/config.json
#RUN wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=1RaWEfzjE9Tx3XbIm_oT2pt0bSJO8J8ox' -O /models/resnet50b_fpn256/archive.tar.gz 
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1RaWEfzjE9Tx3XbIm_oT2pt0bSJO8J8ox' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1RaWEfzjE9Tx3XbIm_oT2pt0bSJO8J8ox" -O /models/resnet50b_fpn256/archive.tar.gz && rm -rf /tmp/cookies.txt
RUN cd /models/resnet50b_fpn256/ && tar -xzf archive.tar.gz && ls
RUN cd /models/resnet50b_fpn256/ && mv Output_Segm_Patch_Resnet50b_fpn256/resnet50b_fpn_final_cpu.pth ./weights.pth && ls
RUN cd /models/resnet50b_fpn256/ && rm archive.tar.gz && ls

RUN chmod 444 /models/resnet50b_fpn256/config.json
RUN chmod 444 /models/resnet50b_fpn256/weights.pth

# ---------------------------------------------------#
# 6. Install scripts and setup entry point
# ---------------------------------------------------#
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

# ---------------------------------------------------#
# 7. Set entrypoint to ">python /app/run.py"
# ---------------------------------------------------#
ENTRYPOINT ["python", "/app/run.py"]