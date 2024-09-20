import streamlit as st

# Set the title of the Streamlit app
st.title("Pre-Experimental Survey")

# Embed the Google Form using the iframe URL
st.components.v1.iframe(
    src="https://docs.google.com/forms/d/e/1FAIpQLScZbUYP52Ucml1tpwIF_1F7V0oSifDjk_5yMUEhh4oqIyTpYA/viewform?embedded=true", 
    width=640, 
    height=1260, 
    scrolling=True
)
