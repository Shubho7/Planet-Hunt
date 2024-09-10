import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib  

# Load the trained model
model = tf.keras.models.load_model('C:/Users/baner/Documents/Planet-Hunt/notebooks/trained_model.h5')  

scaler = joblib.load('C:/Users/baner/Documents/Planet-Hunt/notebooks/scaler.pkl') 

# Streamlit app UI
st.title('Exoplanet Habitability')

# Organizing input fields into two columns
col1, col2 = st.columns(2)

# Column 1 input fields
with col1:
    P_MASS = st.number_input('Mass of the planet')
    P_RADIUS = st.number_input('Planetary Radius of the planet')
    P_PERIOD = st.number_input('Orbital Period of the planet')
    P_SEMI_MAJOR_AXIS = st.number_input('Semi-major Axis of the planet')

# Column 2 input fields
with col2:
   P_ECCENTRICITY = st.number_input('Orbital Eccentricity of the planet')
   P_INCLINATION = st.number_input('Orbital Inclination of the planet')
   P_ESCAPE = st.number_input('Escape Velocity of the planet')
   P_POTENTIAL = st.number_input('Gravitational Potential of the planet')

# Styling the button using st.markdown and HTML
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #1F04FF;
        color:#ffffff;
        border-radius:15px;
        height:50px;
        width:200px;
        font-size:10px;
    }
    div.stButton > button:first-child:hover {
        background-color: #001198;
        color:#ffffff;
    }
    </style>""", unsafe_allow_html=True)

# When the user clicks the Predict button
if st.button('Predict'):
    # Input data from user
    input_data = np.array([[P_MASS, P_RADIUS, P_PERIOD, P_SEMI_MAJOR_AXIS, P_ECCENTRICITY, 
                            P_INCLINATION, P_ESCAPE, P_POTENTIAL]])

    # Scaling the input data
    input_data_scaled = scaler.transform(input_data)

    # Predicting using the model
    prediction = np.argmax(model.predict(input_data_scaled), axis=1)[0]

    # Mapping prediction to the actual class
    if prediction == 0:
        st.success('The planet is Uninhabitable.')
    elif prediction == 1:
        st.success('The planet is Conservatively Habitable.')
    else:
        st.success('The planet is Optimistically Habitable.')
