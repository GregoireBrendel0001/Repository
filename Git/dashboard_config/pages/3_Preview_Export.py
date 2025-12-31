import streamlit as st
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import initialize_session_state, render_preview_export


st.set_page_config(
    page_title="Configuration Preview & Export",
    layout="wide"
)


def main():
    initialize_session_state()
    st.title("Preview & Export")
    render_preview_export()


if __name__ == "__main__":
    main()


