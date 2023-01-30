# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 12:39:42 2022

@author: georgy.peshkov
"""

import streamlit as st

from PIL import Image
import os
from   pathlib import Path
import pandas as pd

#####################################################################################################################


st.markdown("<h1 style='text-align: center; color: black;'>RTFE web application</h1>", unsafe_allow_html=True)

##################################################################################################################### 
st.sidebar.subheader('Uplod data:')
uploaded_file = st.sidebar.file_uploader('Specify the path to the LWD data:')
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.sidebar.write(bytes_data)

st.sidebar.subheader('Data visualization:')
if st.sidebar.button('PLot LWD data'):
    st.subheader('Data visualization')
    image_path = str(Path.joinpath(Path(os.getcwd()).parent, Path("figs") / "3D_resistivity.PNG"))
    MW_image = Image.open(image_path)
    st.image(MW_image)
else:
    st.sidebar.write('')   
    
##################################################################################################################### 
st.sidebar.subheader('Well trajectory planning parameters:')       
if st.sidebar.button('Set', key='1'):
    st.subheader('Well trajectory planning parameters')
    with st.form(key='my_form'):
        X = st.number_input('Well head X [m]', min_value=0.0, max_value=10000000.0, value=0.0, step=0.1)
        Y = st.number_input('Well head Y [m]', min_value=0.0, max_value=10000000.0, value=0.0, step=0.1)
        MD = st.number_input('Planned MD [ft]', min_value=0.0, max_value=36000.0, value=0.0, step=0.1)
        Dog_leg_depth = st.number_input('MD for dogleg starting [ft]', min_value=0.0, max_value=36000.0, value=0.0, step=0.1)
        Dog_leg_az = st.number_input('Azimuth for dogleg starting [ft]', min_value=-180.0, max_value=180.0, value=0.0, step=0.1)
        
        submit_button = st.form_submit_button(label='Submit parameters')    
    
else:
    st.sidebar.write('')    
#####################################################################################################################  
st.sidebar.subheader('RTFE parameters:') 
if st.sidebar.button('Set', key='2'):
    st.subheader('RTFE parameters')
    with st.form(key='my_form2'):
        
        engine = st.selectbox(
            "Choose the engine:",
            ("Greedy", "Q-learning")
            )
        
        correction = st.selectbox(
            "Drilling trajectory correction:",
            ("Automatic", "Manual")
            )
        
        X_RTFE = st.number_input('X [m]', min_value=0.0, max_value=10000000.0, value=0.0, step=0.1)
        Y_RTFE = st.number_input('Y [m]', min_value=0.0, max_value=10000000.0, value=0.0, step=0.1)
        Z_RTFE = st.number_input('Z [m]', min_value=-5000.0, max_value=10000000.0, value=5000.0, step=0.1)
        Az_i_RTFE = st.number_input('Initial azimuth [°]', min_value=-180.0, max_value=180.0, value=0.0, step=0.1)
        Zen_i_RTFE = st.number_input('Initial zenith [°]', min_value=-180.0, max_value=180.0, value=0.0, step=0.1)
        Az_constr = st.number_input('Azimuth constraint [°/10 m]', min_value=0.0, max_value=10.0, value=2.0, step=0.01)
        Zen_constr = st.number_input('Zenith constraint [°/10 m]', min_value=0.0, max_value=10.0, value=2.0, step=0.01)
        DL_constr = st.number_input('Dogleg constraint [°]', min_value=-180.0, max_value=180.0, value=0.0, step=0.1)
        step_L = st.number_input('Length of 1 step [m]', min_value=0.0, max_value=50.0, value=2.0, step=0.1)
        step_ahead = st.number_input('Number of steps ahead', min_value=1, max_value=20, value=2, step=1)
        submit_button = st.form_submit_button(label='Submit')
        
        
#####################################################################################################################  
st.sidebar.subheader('RTFE simulation:')
if st.sidebar.button('Launch'):
    st.subheader('RTFE simulation')
    video_path = str(Path.joinpath(Path(os.getcwd()).parent, Path("Video") / "BHA_Animation_CDF.mp4"))
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
    
else:
    st.sidebar.write('')
    
#####################################################################################################################     
st.sidebar.subheader('Results:')
if st.sidebar.button('Show'):
    
    
    st.subheader('Trajectory 3D view')
    image_path = str(Path.joinpath(Path(os.getcwd()).parent, Path("figs") / "3D_trajectory.PNG"))
    MW_image = Image.open(image_path)
    st.image(MW_image)
    
    st.subheader('Plan view')
    image_path = str(Path.joinpath(Path(os.getcwd()).parent, Path("figs") / "2D_XZ.PNG"))
    MW_image = Image.open(image_path)
    st.image(MW_image)
    
    #st.subheader('Section view')
    #image_path = str(Path.joinpath(Path(os.getcwd()).parent, Path("figs") / "2D_ZY.PNG"))
    #MW_image = Image.open(image_path)
    #st.image(MW_image)
    
    st.subheader('Drilling trajectory table')
    table_path = str(Path.joinpath(Path(os.getcwd()).parent, Path("figs") / "List of corrections.csv"))
    df = pd.read_csv(table_path)
    st.dataframe(df)

    st.subheader('OFV values')
    Plan_tr = 2456
    Corr_tr = 2999
    st.write('Planned trajectory OFV:', Plan_tr)
    st.write('Corrected trajectory OFV:', Corr_tr)