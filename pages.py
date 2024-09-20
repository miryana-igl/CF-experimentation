import streamlit as st
#st.set_page_config(page_title="The Titanic Data Collection and Prediction", page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)


pis_page = st.Page("pis.py", title="Consent Form", icon=":material/add_circle:")
survey_page = st.Page("google_survey.py", title="Survey", icon=":material/add_circle:")
digital = st.Page("digital.py", title="Digital UI", icon=":material/add_circle:")
hybrid = st.Page("hybrid.py", title="Hybrid UI", icon=":material/add_circle:")

pg = st.navigation([pis_page,survey_page,digital, hybrid])
st.set_page_config(page_title="Pages", page_icon=":material/edit:")
pg.run()
