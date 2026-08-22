# Streamlit app for Sydney housing price prediction
import joblib
import pandas as pd
import streamlit as st

MODEL_FILENAME = "housing_price_model.joblib"

# housing_price_model.joblib is built by train_model.py during the Docker image build
model = joblib.load(MODEL_FILENAME)

st.title("Sydney Housing Price Prediction")
st.write(
    "Enter property details to get a predicted sale price. This model only covers "
    "Mosman, Parramatta and Liverpool, the three suburbs it was trained on."
)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    suburb = st.selectbox("Suburb", ["Mosman", "Parramatta", "Liverpool"])
    property_type = st.selectbox("Property Type", ["House", "Townhouse", "Apartment and Unit"])
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=6, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=4, value=1)

with col2:
    car_spaces = st.number_input("Car Spaces", min_value=0, max_value=4, value=1)
    size_m2 = st.number_input(
        "Land size (House/Townhouse) or building size (Apartment) in m²",
        min_value=30, max_value=900, value=150,
    )
    sale_year = st.number_input("Sale Year", min_value=2020, max_value=2030, value=2026)
    sale_month = st.number_input("Sale Month", min_value=1, max_value=12, value=8)

description = st.text_area(
    "Property Description",
    placeholder="Paste the listing description here, e.g. Spacious family home close to shops and transport...",
)

st.markdown("---")

if st.button("Predict Price", type="primary"):
    input_data = pd.DataFrame([{
        "suburb": suburb,
        "property_type": property_type,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "car_spaces": car_spaces,
        "size_m2": size_m2,
        "sale_year": sale_year,
        "sale_month": sale_month,
        "description": description,
    }])

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")
    st.success(f"Predicted Sale Price: ${prediction:,.0f}")
    st.caption(
        "This model was trained on 100 manually collected properties. "
        "Treat this as a guide, not a formal valuation."
    )

st.markdown("---")
st.caption("Powered by a Gradient Boosting model, part of SIG720 Task-8D")
