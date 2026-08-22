# Housing Price Prediction

Streamlit app that predicts Sydney residential sale prices for Mosman, Parramatta and Liverpool, using a Gradient Boosting model trained on 100 manually collected properties. Built for SIG720 Task-8D.

Live app: https://housing-price-prediction-8prt.onrender.com/

## Files

- `2026-08-22-sydney-housing-sold-data.csv`, the training data.
- `train_model.py`, builds the pipeline and saves `housing_price_model.joblib`.
- `app.py`, the Streamlit app.
- `Dockerfile`, builds the model and runs the app.
- `requirements.txt`, Python dependencies.

## Run locally

1. `pip install -r requirements.txt`
2. `python train_model.py`, this creates `housing_price_model.joblib` in the same folder.
3. `streamlit run app.py`, the app opens at `http://localhost:8501`.

## Run with Docker

1. `docker build -t housing-price-prediction .`, this installs dependencies and runs `train_model.py` inside the image automatically.
2. `docker run -p 10000:10000 housing-price-prediction`, the app opens at `http://localhost:10000`.

## Deploy to Render

1. Push this folder to a GitHub repo.
2. On Render.com, create a new Web Service and connect that GitHub repo.
3. Render finds the Dockerfile at the repo root automatically, no extra path configuration is needed.
4. Render builds the Dockerfile (which also builds the model) and starts the container. Every push to the connected branch triggers a new build and deploy automatically.
