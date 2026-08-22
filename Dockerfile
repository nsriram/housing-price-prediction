# Minimal Python base image
FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Build the model as part of the image build, so it is always in sync with this code and data
RUN python train_model.py

# 10000 is Render's default port, Render overrides this if a different port is set in its dashboard
ENV PORT=10000
EXPOSE 10000

# fileWatcherType=none avoids Streamlit's file watcher, which needs more inotify instances
# than Render's containers allow, and is not needed once the app is deployed
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.enableXsrfProtection=false --server.fileWatcherType none
