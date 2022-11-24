# Build step #1: build the React front end
FROM node:lts-alpine as build-step
WORKDIR /app
ENV PATH /app/node_modules/.bin:$PATH
COPY ./src ./src
COPY package.json ./
RUN npm install
COPY ./public ./public
ENV HOST 0.0.0.0
EXPOSE 3000
RUN npm start