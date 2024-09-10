import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib  

# Load the trained model
model = tf.keras.models.load_model('trained_model.h5')  

scaler = joblib.load('scaler.pkl') 

# Streamlit app UI
st.title('Exoplanet Habitability')

# Creating input fields for the 10 features
P_MASS = st.number_input('Planetary Mass (P_MASS)', value=1.0)
P_RADIUS = st.number_input('Planetary Radius (P_RADIUS)', value=1.0)
P_PERIOD = st.number_input('Orbital Period (P_PERIOD)', value=1.0)
P_SEMI_MAJOR_AXIS = st.number_input('Semi-major Axis (P_SEMI_MAJOR_AXIS)', value=1.0)
P_ECCENTRICITY = st.number_input('Orbital Eccentricity (P_ECCENTRICITY)', value=0.0)
P_INCLINATION = st.number_input('Orbital Inclination (P_INCLINATION)', value=0.0)
P_ESCAPE = st.number_input('Escape Velocity (P_ESCAPE)', value=0.0)
P_POTENTIAL = st.number_input('Gravitational Potential (P_POTENTIAL)', value=0.0)
P_GRAVITY = st.number_input('Surface Gravity (P_GRAVITY)', value=0.0)
P_DENSITY = st.number_input('Density (P_DENSITY)', value=1.0)


