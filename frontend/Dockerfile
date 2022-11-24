# Build step #1: build the React front end
FROM node:lts-alpine as build-step
WORKDIR /app
ENV PATH /app/node_modules/.bin:$PATH
COPY package.json ./
RUN npm install
COPY ./src ./src
COPY ./public ./public
ENV HOST=0.0.0.0
RUN npm start

