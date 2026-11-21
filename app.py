"""
Entry point. Run with:  streamlit run app.py

This file only sets up the page, the sidebar experiment picker, and the
navigation. Each page lives in app_pages/ (deliberately not named "pages/", so Streamlit
never auto-discovers it and this file always controls navigation).
"""

import streamlit as st

import ui

st.set_page_config(page_title="A/B Test Decision Engine", page_icon="💎", layout="wide")

PAGES = [
    st.Page("app_pages/home.py", title="Home", icon=":material/home:", default=True),
    st.Page("app_pages/trust.py", title="Trust checks", icon=":material/verified_user:"),
    st.Page("app_pages/results.py", title="Results", icon=":material/bar_chart:"),
    st.Page("app_pages/segments.py", title="Segments", icon=":material/groups:"),
    st.Page("app_pages/uplift.py", title="Uplift targeting", icon=":material/target:"),
    st.Page("app_pages/analyst.py", title="AI Analyst", icon=":material/auto_awesome:"),
    st.Page("app_pages/plan.py", title="Plan a test", icon=":material/calculate:"),
    st.Page("app_pages/how_it_works.py", title="How it works", icon=":material/menu_book:"),
]

nav = st.navigation(PAGES)
ui.experiment_picker()   # sidebar picker shared by every page
nav.run()
