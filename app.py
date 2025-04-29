import streamlit as st
import pandas as pd
import numpy as np
# import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Projectile Motion Predictor", layout="centered")


# --- Load and train model (or load a pre-trained model) ---
@st.cache_resource
def load_model():
    df = pd.read_csv("projectile_dataset.csv")
    X = df[['v0', 'sin_theta', 'cos_theta', 'U', 'sin_alpha', 'cos_alpha']]
    y = df[['Range', 'Height', 'FlightTime']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    return model

model = load_model()

st.title("Projectile Trajectory Predictor Under Wind Influence")
st.write("Enter the launch conditions to predict projectile motion characteristics.")

# --- User Input ---
v0_input = st.number_input("Initial Speed v₀ (m/s)", min_value=1.0, max_value=200.0, value=50.0)
theta_input = st.slider("Launch Angle θ (degrees)", min_value=0.0, max_value=90.0, value=45.0)
U_input = st.slider("Wind Speed U (m/s)", min_value=0.0, max_value=50.0, value=10.0)
alpha_input = st.slider("Wind Direction α (degrees)", min_value=0.0, max_value=360.0, value=180.0)

# --- Preprocess Input ---
sin_theta = np.sin(np.radians(theta_input))
cos_theta = np.cos(np.radians(theta_input))
sin_alpha = np.sin(np.radians(alpha_input))
cos_alpha = np.cos(np.radians(alpha_input))

input_features = np.array([[v0_input, sin_theta, cos_theta, U_input, sin_alpha, cos_alpha]])

# --- Make Prediction ---
if st.button("Predict Trajectory"):
    predicted_outputs = model.predict(input_features)[0]
    
    predicted_range = max(predicted_outputs[0], 0)
    predicted_height = max(predicted_outputs[1], 0)
    predicted_time_of_flight = max(predicted_outputs[2], 0)

    # --- Display Results ---
    st.markdown("### Predicted Trajectory")
    st.metric("Range", f"{predicted_range:.2f} m")
    st.metric("Max Height", f"{predicted_height:.2f} m")
    st.metric("Time of Flight", f"{predicted_time_of_flight:.2f} s")

    # Optional Plot
    import matplotlib.pyplot as plt
    t = np.linspace(0, predicted_time_of_flight, 300)
    g = 9.81
    km = 0.25
    vx = v0_input * cos_theta + U_input * cos_alpha
    vy0 = v0_input * sin_theta - U_input * sin_alpha
    y = (vy0 / km) * (1 - np.exp(-km * t)) - (g / km) * t
    x = (vx / km) * (1 - np.exp(-km * t))

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Projectile Trajectory")
    ax.grid(True)
    st.pyplot(fig)
