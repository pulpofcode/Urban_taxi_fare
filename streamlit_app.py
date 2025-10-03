# -----------------------------
# streamlit_app.py
# -----------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import os

# -----------------------------
# Load Trained Model
# -----------------------------

model = joblib.load("D:/Python/ML_Projects/Urban_taxi_fare/Models/best_taxi_fare_model.pkl")
#model = joblib.load(os.path.join("Models", "best_taxi_fare_model.pkl"))

# -----------------------------
# Helper Functions
# -----------------------------
def haversine_np(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance between two points in km."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def preprocess_input(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
                     passenger_count, payment_type, pickup_datetime):
    """Preprocess user input into model-ready features."""

    df = pd.DataFrame({
        'passenger_count':[passenger_count],
        'pickup_latitude':[pickup_lat],
        'pickup_longitude':[pickup_lon],
        'dropoff_latitude':[dropoff_lat],
        'dropoff_longitude':[dropoff_lon],
        'payment_type':[payment_type]
    })

    # Trip distance
    df['trip_distance'] = haversine_np(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )

    # Trip duration approximation (assuming avg speed ~30 km/h)
    df['trip_duration_min'] = df['trip_distance'] / 0.5

    # Time-based features
    df['pickup_hour'] = pickup_datetime.hour
    df['am_pm'] = 'AM' if pickup_datetime.hour < 12 else 'PM'

    # Encode categorical variables
    df = pd.get_dummies(df, columns=['am_pm', 'payment_type'], drop_first=True)

    # Ensure model's feature columns exist
    model_cols = model.feature_names_in_
    for col in model_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[model_cols]

    return df, df['trip_distance'].values[0]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🚕 TripFare: Urban Taxi Fare Prediction")
st.markdown("""
Predict taxi fares within **New York City**.
- **Latitude Range:** 40.5°N – 40.9°N  
- **Longitude Range:** -74.25°W – -73.7°W
""")
st.markdown("Enter trip details below to predict the total fare amount.")

# -----------------------------
# Inputs
# -----------------------------
pickup_lat = st.number_input(
    "Pickup Latitude (NYC approx: 40.5774°N - 40.9176°N)", 
    value=40.7614327, min_value=40.5774, max_value=40.9176
)
pickup_lon = st.number_input(
    "Pickup Longitude (NYC approx: -74.15°W - -73.7004°W)", 
    value=-73.9798156, min_value=-74.15, max_value=-73.7004
)
dropoff_lat = st.number_input(
    "Dropoff Latitude (NYC approx: 40.5774°N - 40.9176°N)", 
    value=40.6513111, min_value=40.5774, max_value=40.9176
)
dropoff_lon = st.number_input(
    "Dropoff Longitude (NYC approx: -74.15°W - -73.7004°W)", 
    value=-73.8803331, min_value=-74.15, max_value=-73.7004
)
passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, value=1)
payment_type = st.selectbox("Payment Type", ["CRD", "CSH", "NOC", "DIS", "UNK"])

# -----------------------------
# Date + Time Inputs
# -----------------------------
now = datetime.datetime.now()
pickup_date = st.date_input("Pickup Date", min_value=now.date())
pickup_time = st.time_input("Pickup Time", value=datetime.time(now.hour, now.minute))
pickup_datetime = datetime.datetime.combine(pickup_date, pickup_time)

# -----------------------------
# Past Datetime Warning
# -----------------------------
if pickup_datetime < now:
    st.warning("⚠️ Pickup datetime cannot be in the past!")

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict Fare"):
    if pickup_datetime < now:
        st.error("Cannot predict fare for past pickup datetime.")
    else:
        X_input, distance_km = preprocess_input(
            pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
            passenger_count, payment_type, pickup_datetime
        )
        prediction = model.predict(X_input)[0]
        st.success(f"Predicted Total Fare: ${prediction:.2f}")
        st.info(f"Trip Distance: {distance_km:.2f} km")
