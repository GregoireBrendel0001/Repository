import streamlit as st
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import initialize_session_state, render_conditions_config


st.set_page_config(
    page_title="Trading Conditions Builder",
    layout="wide"
)


def main():
    initialize_session_state()
    st.title("Buy & Sell Conditions")
    render_conditions_config()


if __name__ == "__main__":
    main()
