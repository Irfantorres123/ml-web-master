### user:

run `docker-compose up`

go to http://localhost

#### mac users:
If you see `operation not permitted` - go to System preferences > Security & Privacy > Files and Folders, and add Docker for Mac and your shared directory.

### existing docker containers:
- https://hub.docker.com/r/hbao34/ml-web-frontend
- https://hub.docker.com/r/hbao34/ml-web-backend

#### build your own docker container:
first you log in with:
`docker login`

then build on local with docker build:
`docker build -t <your_username>/<container_name>:<tag_name> .`

test your build:
`docker run <your_username>/<container_name>:<tag_name>`

finally push:
`docker push <your_username>/<container_name>:<tag_name>`

### to deploy on server: 
make your own docker containers, update `docker-compose.yml` with new containers

upload `docker-compose.yml` and `Caddyfile` to server and run `docker-compose up`

### develop on local:
```
cd backend
pip install --no-cache-dir --upgrade -r requirements.txt
python app/main.py

cd ../frontend
npm install
npm run build
npm start

go to http://localhost:3000
```

### file tree
```
key files:
.
├── frontend
│   ├── src
│   │   ├── resultTable.js
│   │   ├── upload.js
│   │   ├── setupProxy.js
│   │   ├── App.js
│   │   ├── index.js
│   │   └── index.css
│   └── Dockerfile
├── backend
│   ├── app
│   │   ├── data
│   │   ├── main.py
│   │   ├── api.py // change model here
│   │   ├── generate_curve_spectrum_one_img.py // scoliosis pipline
│   │   ├── preprocess_one_img
│   │   ├── latent_space_spectrum.pth // large file, get it here: https://drive.google.com/file/d/1GmNCNTAAaKGb-cRGa8BHYb6of7PTq5l1/view?usp=sharing
│   │   ├── spectrum_angle.pth
│   │   └── unet.pth
│   └── Dockerfile
├── scoliosis_images
├── mnist_images
├── docker-compose.yml
├── Caddyfile
└── README.md
```
