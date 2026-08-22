# Builds the same pipeline recommended in Part 3 and saves it as a joblib file.
# This runs automatically during the Docker image build, see the Dockerfile.
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

MODEL_FILENAME = "housing_price_model.joblib"

FEATURE_COLS = [
    "suburb", "property_type", "bedrooms", "bathrooms", "car_spaces",
    "size_m2", "sale_year", "sale_month", "description",
]
TARGET = "sale_price_aud"

# Suburb names and property type words are excluded from the text vocabulary,
# they would just duplicate the suburb and property_type columns, same as in the notebook
LEAK_WORDS = ["mosman", "parramatta", "liverpool", "apartment", "unit", "house", "townhouse"]
TEXT_STOP_WORDS = list(ENGLISH_STOP_WORDS) + LEAK_WORDS


def load_data():
    df = pd.read_csv("2026-08-22-sydney-housing-sold-data.csv")
    df["sale_date"] = pd.to_datetime(df["sale_date"], format="%d-%b-%Y")
    df["sale_year"] = df["sale_date"].dt.year
    df["sale_month"] = df["sale_date"].dt.month
    df["size_m2"] = df["land_size_m2"].fillna(df["building_siz_m2"])
    return df


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("suburb_type", OneHotEncoder(handle_unknown="ignore"), ["suburb", "property_type"]),
            ("text", TfidfVectorizer(stop_words=TEXT_STOP_WORDS, min_df=3, max_features=15), "description"),
        ],
        remainder="passthrough",
    )
    # Gradient Boosting was the model recommended in Part 3
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(random_state=42)),
    ])


def main():
    df = load_data()
    X = df[FEATURE_COLS]
    y = df[TARGET]

    model = build_pipeline()

    # Fit on every available row. The notebook's train/validation/test split was only
    # needed for honest evaluation, the deployed model should learn from all of it.
    model.fit(X, y)
    joblib.dump(model, MODEL_FILENAME)
    print(f"Model saved to {MODEL_FILENAME}")


if __name__ == "__main__":
    main()
