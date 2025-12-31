import streamlit as st
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import initialize_session_state, render_configuration_form


st.set_page_config(
    page_title="Strategy Configuration Dashboard",
    layout="wide"
)


def main():
    initialize_session_state()
    st.title("Strategy Configuration")
    render_configuration_form()


if __name__ == "__main__":
    main()
