import os
import streamlit as st
import pickle

def file_selector(folder_path, on_change):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames, on_change= on_change)

    return os.path.join(folder_path, selected_filename)