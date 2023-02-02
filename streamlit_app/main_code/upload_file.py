import streamlit as st
import os
import pickle

def save_uploadedfile(uploadedfile):
    with open(os.path.join("streamlit_app/data/raw/" ,uploadedfile.name) ,"wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} to server".format(uploadedfile.name))


def upload_selected_file(path):
    with open(os.path.join(path),"rb") as f:
        file = pickle.load(f)
    return file