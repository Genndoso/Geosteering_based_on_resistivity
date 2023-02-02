import os
import streamlit as st
import pickle

def file_selector(folder_path):
    filenames = os.listdir(folder_path)
    selected_filename = st.sidebar.selectbox('Select a file', filenames)

    return os.path.join(folder_path, selected_filename)