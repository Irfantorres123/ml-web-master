FROM python:3.9
WORKDIR /backend
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1
COPY ./requirements.txt /backend/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /backend/requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python
RUN pip install scipy
RUN pip install scikit-image
RUN pip install numpy
RUN pip install torchvision
RUN pip install opencv-python
COPY ./app /backend/app
EXPOSE 8000
CMD ["python", "./app/main.py"]