# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 12:39:42 2022

@author: georgy.peshkov
"""

import streamlit as st

from PIL import Image
import os
from  pathlib import Path
import pandas as pd
import pickle
from main_code.upload_file import save_uploadedfile, upload_selected_file
from main_code.file_selector import file_selector
#####################################################################################################################
# #load mages for examples
# XY_path = str(Path.joinpath(Path(os.getcwd()).parent, Path("data") / "images/2D_XY.PNG"))
# XY_image = Image.open(XY_path)
#
#
# YZ_path = str(Path.joinpath(Path(os.getcwd()).parent, Path("data") / "images/2D_ZY.PNG"))
# YZ_image = Image.open(YZ_path)
#
#
# XZ_path = str(Path.joinpath(Path(os.getcwd()).parent, Path("data") / "images/2D_XZ.PNG"))
# XZ_image = Image.open(XZ_path)

#####################################################################################################################
folder_path = os.path.join("data/raw/")

st.markdown("<h1 style='text-align: center; color: black;'> Real time formation evaluation based on resistivity</h1>", unsafe_allow_html=True)

##################################################################################################################### 
st.markdown(' **Upload or select existing data:**')
type_of_use = st.radio('', ['Upload file', 'Select from existing'])

if type_of_use == 'Upload file':
    uploaded_file = st.file_uploader('Specify the path to the LWD data:')
    if uploaded_file is not None:
        file = pickle.load(uploaded_file)
        save_uploadedfile(uploaded_file)

elif type_of_use == 'Select from existing':

    path = file_selector(folder_path)
    file = upload_selected_file(path)



st.sidebar.subheader('Data visualization:')
if st.sidebar.button('PLot LWD data'):
    st.subheader('Data visualization')
    
    vis_type = st.selectbox("Choose type of plotting:", [ "2D",
                                                          "3D"])
    
    if vis_type == "2D":
        
        x_y_z_vis = st.selectbox("Choose a plane for plotting:", [  "X-Y",
                                                                    "Y-Z",
                                                                    "X-Z"])
        
        column1, column2 = st.columns([1, 4])
        
        if x_y_z_vis == "X-Y":          
            # Add slider to column 1
            slider = column1.slider("Slice # along X-Y plane", min_value=0, max_value=400, value=1)
            # Add plot to column 2
            column2.image(XY_image, width=500)
        
        elif x_y_z_vis == "Y-Z":
            # Add slider to column 1
            slider = column1.slider("Slice # along Y-Z plane", min_value=0, max_value=1000, value=1)
            # Add plot to column 2
            column2.image(YZ_image, width=500)
            
        elif x_y_z_vis == "X-Z":          
            # Add slider to column 1
            slider = column1.slider("Slice # along Y-Z plane", min_value=0, max_value=400, value=1)
            # Add plot to column 2
            column2.image(XZ_image, width=500)
       
    elif vis_type == "3D":     
        # Create two columns
        column1, column2 = st.columns([1, 4])
        
        # Add sliders to column 1
        slider1 = column1.slider("Slice along X-Y plane", min_value=0, max_value=400, value=1)
        slider2 = column1.slider("Slice along Y-Z axis", min_value=0, max_value=1000, value=1)
        slider3 = column1.slider("Slice along X-Z axis", min_value=0, max_value=400, value=1)
        
        # Add plot to column 2
        image_path = str(Path.joinpath(Path(os.getcwd()).parent, Path("figs") / "3D_resistivity.PNG"))
        image = Image.open(image_path)
        column2.image(image, width=500)
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