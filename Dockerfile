FROM python:3.9.5
RUN pip install --upgrade pip
RUN pip install tensorflow 
RUN pip install scikit-learn
RUN pip install opencv-python
RUN pip install nptyping
RUN pip install numba
RUN apt-get update
RUN apt-get install -y python3-opencv
RUN pip install matplotlib
WORKDIR /work

CMD ["python","main.py"]


