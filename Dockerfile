FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN apt-get update && apt-get install -y python3-tk xvfb

# minimal Dockerfile with tk support

ENV DISPLAY=:99
RUN Xvfb :99 -screen 0 1024x768x24 &
