FROM node:14.15.3-alpine3.12 as client

# Working directory be app
WORKDIR /usr/src/app/client/

COPY frontend/package*.json ./

# Install dependencies
RUN npm install

# copy local files to app folder
COPY frontend/ ./

RUN npm run build


FROM python:3.9
WORKDIR /usr/src/app
COPY --from=client /usr/src/app/client/build/ ./client/build/
WORKDIR /usr/src/app/server/
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1
COPY backend/requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python
RUN pip install scipy
RUN pip install scikit-image
RUN pip install numpy
RUN pip install torchvision
RUN pip install opencv-python
COPY backend/app ./app

CMD ["python", "./app/main.py"]