# Minimal Python base image
FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Build the model as part of the image build, so it is always in sync with this code and data
RUN python train_model.py

# Render sets its own PORT at runtime, 8501 is only the local default
ENV PORT=8501
EXPOSE 8501

CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.enableXsrfProtection=false
