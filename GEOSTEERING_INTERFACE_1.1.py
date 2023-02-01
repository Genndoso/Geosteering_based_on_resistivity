# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 12:39:42 2022

@author: georgy.peshkov
"""

import streamlit as st
from pathlib import Path
from PIL import Image
import os
#os.chdir(Path(os.getcwd()).parent)
print(os.getcwd())
import pandas as pd
import pickle
from streamlit_app.main_code.upload_file import save_uploadedfile, upload_selected_file
from streamlit_app.main_code.file_selector import file_selector
from streamlit_app.main_code.visualize import visualize_cube

from Algorithm.geosteering_3d.differential_evolution.diff_evolution import DE_algo, plot_results




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
#st.write(Path(os.getcwd()))
#####################################################################################################################
folder_path = str(Path("streamlit_app/data/raw"))

st.markdown("<h1 style='text-align: center; color: black;'> Real time formation evaluation based on resistivity</h1>", unsafe_allow_html=True)

##################################################################################################################### 
st.markdown(' **Upload or select existing data:**')
type_of_use = st.radio('', ['Upload file', 'Select from existing'])
st.session_state.counter = 0
if type_of_use == 'Upload file':
    uploaded_file = st.file_uploader('Specify the path to the LWD data:')
    if uploaded_file is not None:
        file = pickle.load(uploaded_file)
        save_uploadedfile(uploaded_file)

elif type_of_use == 'Select from existing':

    path = file_selector(folder_path)
    st.session_state['file_path'] = path
    file = upload_selected_file(st.session_state['file_path'])


st.sidebar.subheader('Data visualization:')
plotter = st.sidebar.button('PLot LWD data', disabled= False)
if plotter :
    try:
        st.sidebar.subheader('Data visualization')

        vis_type = st.radio("Choose type of plotting:", ["2D", "3D"])
        st.session_state.counter += 1
        if vis_type == "2D":
             slider = st.slider("Slice # along X-Y plane", min_value=0.0, max_value=1.0, step=0.1)
             st.write(slider)

        elif vis_type == "3D":
            file = upload_selected_file(st.session_state['file_path'])
            visualize_cube(file, opacity = 1)
    except:
        st.write('Upload data file')
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
global type_of_engine
st.sidebar.subheader('RTFE parameters:')
if st.sidebar.button('Set', key='2'):
    st.subheader('RTFE parameters')
    with st.form(key='my_form2'):
        
        type_of_engine = st.selectbox(
            "Choose the engine:",
            ("Evolution algorithm", "Reinforcement learning")
            )

        if 'engine' not in st.session_state:
            st.session_state['engine'] = type_of_engine

        # correction = st.selectbox(
        #     "Drilling trajectory correction:",
        #     ("Automatic", "Manual")
        #     )
        
        st.session_state['X_RTFE'] = st.number_input('X [m]', min_value=0.0, max_value=10000000.0, value=100.0, step=0.1)
        st.session_state['Y_RTFE'] = st.number_input('Y [m]', min_value=0.0, max_value=10000000.0, value=100.0, step=0.1)
        st.session_state['Z_RTFE'] = st.number_input('Z [m]', min_value=-5000.0, max_value=10000000.0, value=100.0, step=0.1)
        st.session_state['Az_i_RTFE'] = st.number_input('Initial azimuth [°]', min_value=-180.0, max_value=180.0, value=0.0, step=0.1)
        st.session_state['Zen_i_RTFE'] = st.number_input('Initial zenith [°]', min_value=0.0, max_value=180.0, value=0.0, step=0.1)
        st.session_state['Az_constr'] = st.number_input('Azimuth constraint [°]', min_value=0.0, max_value=180.0, value=180.0, step=0.01)
        st.session_state['Zen_constr'] = st.number_input('Zenith constraint [°]', min_value=0.0, max_value=92.0, value=92.0, step=0.01)
        st.session_state['DL_constr'] = st.number_input('Dogleg constraint [°]', min_value=-180.0, max_value=180.0, value=0.0, step=0.1)
        st.session_state['Step_L'] = st.number_input('Length of 1 step [m]', min_value=0.0, max_value=50.0, value=2.0, step=1.0)
       # step_ahead = st.number_input('Number of steps ahead', min_value=1, max_value=20, value=2, step=1)
       #  if st.session_state['engine'] == 'Evolution algorithm':
       #       engine = DE_algo(file)

        submit_button = st.form_submit_button(label='Submit')


#####################################################################################################################  
st.sidebar.subheader('RTFE simulation:')

if st.sidebar.button('Launch'):
    st.subheader('RTFE simulation')

    if st.session_state['engine'] == 'Evolution algorithm':
        cube = upload_selected_file(st.session_state['file_path'])
        engine = DE_algo(cube)
        print([st.session_state['X_RTFE'], st.session_state['Y_RTFE'], st.session_state['Z_RTFE']])


        OFV, traj, df = engine.DE_planning(

            bounds=[(-20, st.session_state['Az_constr']), (0, st.session_state['Zen_constr'] )],
            length=st.session_state['Step_L'], angle_constraint=st.session_state['DL_constr'],
            init_incl=[st.session_state['Zen_i_RTFE'], st.session_state['Zen_i_RTFE']],
            init_azi=[st.session_state['Az_i_RTFE'], st.session_state['Az_i_RTFE']],)
         #  init_pos=[st.session_state['X_RTFE'], st.session_state['Y_RTFE'], st.session_state['Z_RTFE']])


        st.success(f'Objective function value : {round(float(OFV),2)}')

        st.session_state['ready_engine'] = engine
        st.session_state['trajectory'] = traj
        st.session_state['OFV'] = round(float(OFV),2)
        st.session_state['trajectory_table'] = df
        # video_path = str(Path.joinpath(Path(os.getcwd()).parent, Path("Video") / "BHA_Animation_CDF.mp4"))
        # video_file = open(video_path, 'rb')
        # video_bytes = video_file.read()
        # st.video(video_bytes)

    #st.write('Assign RTFE parameters into algorithm')
    
else:
    st.sidebar.write('')
    
#####################################################################################################################     
st.sidebar.subheader('Results:')
if st.sidebar.button('Show'):
    st.write('')
    
    st.subheader('Trajectory 3D view')
    st.write('Later')
    
    st.subheader('Plan view')
    cube = upload_selected_file(st.session_state['file_path'])

    fig = plot_results(cube,st.session_state['trajectory'][0], st.session_state['trajectory'][1],
                 st.session_state['trajectory'][2] )
    st.pyplot(fig)
    #st.subheader('Section view')
    #image_path = str(Path.joinpath(Path(os.getcwd()).parent, Path("figs") / "2D_ZY.PNG"))
    #MW_image = Image.open(image_path)
    #st.image(MW_image)
    
    st.subheader('Drilling trajectory table')
    st.dataframe(st.session_state['trajectory_table'])

    st.subheader('OFV values')
    Plan_tr = 2456

    st.write('Planned trajectory OFV:', Plan_tr)
    st.write('Corrected trajectory OFV:', st.session_state['OFV'])