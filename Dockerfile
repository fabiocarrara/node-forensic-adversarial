# FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

RUN apt update && apt install -y git
ADD requirements.txt .
RUN pip install -r requirements.txt