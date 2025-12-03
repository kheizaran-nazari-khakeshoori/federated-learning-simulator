FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN apt-get update && apt-get install -y python3-tk xvfb
