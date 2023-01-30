# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 18:25:09 2023

@author: Максимилиан
"""
import streamlit as st
import pickle
st.write("Geosteering based on resistivity")

uploaded_file = st.file_uploader(label = 'Upload resistivity cube. It must be a  ')

if uploaded_file is not None:
    file = pickle.load(uploaded_file)
    st.write(f'type of the file {type(file)}')
