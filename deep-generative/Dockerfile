FROM  python:latest
from tensorflow/tensorflow:latest
from tensorflow/tensorflow:devel-gpu

#RUN apt update
COPY . .
RUN ls -l
#RUN apt install python3-dev python3-pip python3-venv
#RUN python3 -m venv --system-site-packages ./venv
#RUN source ./venv/bin/activate
RUN pip install --upgrade pip
RUN pip install -r requirements.txt 
RUN pip install tensorflow==2.1.0
#ENTRYPOINT python3 dcgan.py ; python3 vae.py
ENTRYPOINT python3 vae.py ; python3 dcgan.py 
