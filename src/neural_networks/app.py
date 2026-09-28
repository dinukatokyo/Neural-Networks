import streamlit as st

st.title("Neural Network Demo")

st.write("Streamlit is working!")

study_hours = st.slider(
    "Study Hours per Day",
    min_value=0.0,
    max_value=12.0,
    value=5.0
)

st.write(f"Selected study hours: {study_hours}")