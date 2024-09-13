import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib  

# Load the trained model
model = tf.keras.models.load_model('C:/Users/baner/Documents/Planet-Hunt/notebooks/trained_model.h5')  

scaler = joblib.load('C:/Users/baner/Documents/Planet-Hunt/notebooks/scaler.pkl') 

# Streamlit app UI
# Tile
st.markdown("""
    <h2 style="text-align: center; 
            color: white;">
            <b>PLANET HUNT</b></h2>
""", unsafe_allow_html=True)

# Description
st.markdown("""
    <div style="padding:15px;">
        <p style="font-style: Arial; 
            font-size:18px; 
            color:#ffffff; 
            text-align: justify;">
            Welcome to <b>Planet Hunt</b>🪐✨ an exoplanet habitability predictor<br>
            Imagine looking up at the night sky and knowing that out there beyond the twinkling stars, lie countless hidden worlds—planets with wild landscapes, alien weather, and maybe, just maybe, the right ingredients for life. These are Exoplanets, and we’re on a mission to uncover their secrets. Some may be scorching hot giants, others frozen, but a few could be habitable oases in the vast cosmic desert drifting in the vast emptiness of space. Whenever we spot a new exoplanet, it feels like opening a door to a new sci-fi story. Some could be potential homes for life, with alien sunsets we can only dream of. Who knows, the next big discovery might just reveal a second Earth?, or a world so strange it redefines our imagination. Let’s find out together!
        </p>
    </div>
""", unsafe_allow_html=True)

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
        background-color:#1F04FF;
        color:#ffffff;
        border-radius:10px;
        font-size:100px;
        padding:10px;
        width:100%;
    }
    div.stButton > button:first-child:hover {
        background-color: #001198;
        color:#ffffff;
    }
    </style>""", unsafe_allow_html=True)

# When the user clicks the Predict button
if st.button('PREDICT'):

    # Input data from user
    input_data = np.array([[P_MASS, P_RADIUS, P_PERIOD, P_SEMI_MAJOR_AXIS, P_ECCENTRICITY, 
                            P_INCLINATION, P_ESCAPE, P_POTENTIAL]])

    # Scaling the input data
    input_data_scaled = scaler.transform(input_data)

    # Predicting using the model
    prediction = np.argmax(model.predict(input_data_scaled), axis=1)[0]

    # Displaying the prediction
    if prediction == 0:
        # Uninhabitable
        st.markdown(f"""
            <div style="background-color:#f20000;
                padding-left:10px;
                padding-top:12px;
                border-radius:10px;
                display:inline-block;">
                <h5 style="color:white;
                text-align:left;">
                Oh no! you might have found an <b>Uninhabitable Planet</b></h5>
            </div>
        """, unsafe_allow_html=True)
    elif prediction == 1:
        # Conservatively Habitable
        st.markdown(f"""
            <div style="background-color:#00D032;
                padding-left:10px;
                padding-top:12px;
                border-radius:10px;
                display:inline-block;">
                <h5 style="color:white;
                text-align:left;">
                Whoa! you might have just found a <b>Conservatively Habitable Planet</b></h5>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Optimistically Habitable
        st.markdown(f"""
            <div style="background-color:#FFAF0B;
                padding-left:10px;
                padding-top:12px;
                border-radius:10px;
                display:inline-block;">
                <h5 style="color:black;
                text-align:left;">
                Keep going! you might have found a <b>Optimistically Habitable Planet</b></h5>
            </div>
        """, unsafe_allow_html=True)
