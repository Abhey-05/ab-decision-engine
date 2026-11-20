"""
Entry point. Run with:  streamlit run app.py

This file only sets up the page, the sidebar experiment picker, and the
navigation. Each page lives in pages/.
"""

import streamlit as st

import ui

st.set_page_config(page_title="A/B Test Decision Engine", page_icon="💎", layout="wide")

PAGES = [
    st.Page("pages/home.py", title="Home", icon=":material/home:", default=True),
    st.Page("pages/trust.py", title="Trust checks", icon=":material/verified_user:"),
    st.Page("pages/results.py", title="Results", icon=":material/bar_chart:"),
    st.Page("pages/segments.py", title="Segments", icon=":material/groups:"),
    st.Page("pages/uplift.py", title="Uplift targeting", icon=":material/target:"),
    st.Page("pages/analyst.py", title="AI Analyst", icon=":material/auto_awesome:"),
    st.Page("pages/plan.py", title="Plan a test", icon=":material/calculate:"),
    st.Page("pages/how_it_works.py", title="How it works", icon=":material/menu_book:"),
]

nav = st.navigation(PAGES)
ui.experiment_picker()   # sidebar picker shared by every page
nav.run()
