# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 12:39:42 2022
# streamlit run GEOSTEERING_INTERFACE_1.1.py --server.enableCORS=false


@author: georgy.peshkov
"""
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from PIL import Image
import os
#os.chdir(Path(os.getcwd()).parent)
import pandas as pd
import pickle
from streamlit_app.main_code.upload_file import save_uploadedfile, upload_selected_file
from streamlit_app.main_code.file_selector import file_selector
from streamlit_app.main_code.visualize import visualize_cube, traj_plot

from Algorithm.geosteering_3d.differential_evolution.diff_evolution import DE_algo, plot_results

if 'button_clicked_data' not in st.session_state:
    st.session_state['button_clicked_data'] = False
    st.session_state['button_clicked'] = False
    st.session_state['button_clicked'] = False
    st.session_state['button_set_clicked'] = False
    st.session_state['button_launch_clicked'] = False

def callback_data():
    st.session_state.button_clicked_data = True
    st.session_state.button_clicked = False
    st.session_state.type_of_load = 'Select from existing file'
    st.session_state.button_set_clicked = False
    st.session_state.button_launch_clicked = False

def callback():
    st.session_state.button_clicked_data = False
    st.session_state.button_clicked = True
    st.session_state.vis_type = '3D'
    st.session_state.projection = 'X-Y'
    st.session_state.button_set_clicked = False
    st.session_state.button_launch_clicked = False

def callback_set():
    st.session_state.button_clicked_data = False
    st.session_state.button_clicked = False
    st.session_state.button_set_clicked = True
    st.session_state.button_launch_clicked = False


def no_vis():
    st.session_state.button_clicked = False


#####################################################################################################################
folder_path = str(Path("streamlit_app/data/raw"))

st.markdown("<h1 style='text-align: center; color: black;'> Real time formation evaluation based on resistivity</h1>", unsafe_allow_html=True)

##################################################################################################################### 

###########################################################################
st.sidebar.subheader('Data')

st.session_state['data_button'] = st.sidebar.button('Load data', on_click=callback_data)
if st.session_state.button_clicked_data:
    st.markdown(' **Upload or select existing data:**')
    st.session_state['type_of_load'] = st.radio('', ['Upload file', 'Select from existing file'],
                                                on_change= callback_data)


    if st.session_state['type_of_load'] == 'Upload file':
        uploaded_file = st.file_uploader('Specify the path to the LWD data:', on_change= callback_data)
        if uploaded_file is not None:

            st.session_state.file = pickle.load(uploaded_file)
            save_uploadedfile(uploaded_file)


    elif st.session_state['type_of_load'] == 'Select from existing file':

        path = file_selector(folder_path, on_change = callback_data)
        st.session_state['file_path'] = path
        st.session_state.file = upload_selected_file(st.session_state['file_path'])
        st.success("Dataset {} is successfully loaded".format((path.split('/')[-1])))

###########################################################################
st.sidebar.subheader('Data visualization:')
st.session_state['plot_button'] = st.sidebar.button('PLot LWD data', disabled= False, on_click= callback)
if st.session_state.button_clicked:


        st.session_state['vis_type'] = st.radio("Choose type of plotting:", ["2D", "3D"], on_change = callback)
        if st.session_state['vis_type'] == "2D":
        ############################################################

         #   st.session_state.file = upload_selected_file(st.session_state['file_path'])

            st.session_state.projection = st.selectbox("Choose a plane for plotting:", ["X-Y",
                                                                      "Y-Z",
                                                                      "X-Z"], on_change=callback)
            column1, column2 = st.columns([1, 4])

            if st.session_state.projection == "X-Y":
                # Add slider to column 1
                slider = column1.slider("Slice # along X-Y plane", min_value=0, max_value=st.session_state.file.shape[2],
                                      on_change=callback, value=1)
                # Add plot to column 2
                fig, ax = plt.subplots(1, 1)
                ax = plt.imshow(st.session_state.file[:, :, slider])
                st.pyplot(fig)

            elif st.session_state.projection == "Y-Z":
                # Add slider to column 1
                slider = column1.slider("Slice # along Y-Z plane", min_value=0,
                                        on_change=callback, max_value=st.session_state.file.shape[0], value=1)
                # Add plot to column 2
                fig, ax = plt.subplots(1, 1)
                ax = plt.imshow(st.session_state.file[slider, :, :])
                st.pyplot(fig)

            elif st.session_state.projection == "X-Z":
                # Add slider to column 1
                slider = column1.slider("Slice # along X-Z plane", min_value=0,
                                        on_change=callback, max_value=st.session_state.file.shape[1], value=1)
                # Add plot to column 2
                fig, ax = plt.subplots(1,1)
                ax = plt.imshow(st.session_state.file[:, slider, :])
                st.pyplot(fig)


        #################################################################################

        elif st.session_state['vis_type'] == "3D":
            op_slider = st.slider("Opacity", min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                                  key='opacity_slider')

          #  st.session_state.file = upload_selected_file(st.session_state['file_path'])
            visualize_cube(st.session_state.file, opacity = op_slider)

    #    st.write('Upload data file')
else:
    st.sidebar.write('')
    
##################################################################################################################### 
# st.sidebar.subheader('Well trajectory planning parameters:')
# if st.sidebar.button('Set', key='1', on_click=no_vis):
#     st.subheader('Well trajectory planning parameters')
#     with st.form(key='my_form'):
#         X = st.number_input('Well head X [m]', min_value=0.0, max_value=10000000.0, value=0.0, step=0.1)
#         Y = st.number_input('Well head Y [m]', min_value=0.0, max_value=10000000.0, value=0.0, step=0.1)
#         MD = st.number_input('Planned MD [ft]', min_value=0.0, max_value=36000.0, value=0.0, step=0.1)
#         Dog_leg_depth = st.number_input('MD for dogleg starting [ft]', min_value=0.0, max_value=36000.0, value=0.0, step=0.1)
#         Dog_leg_az = st.number_input('Azimuth for dogleg starting [ft]', min_value=-180.0, max_value=180.0, value=0.0, step=0.1)
#
#         submit_button = st.form_submit_button(label='Submit parameters')
#
# else:
#     st.sidebar.write('')
#####################################################################################################################  


st.sidebar.subheader('RTFE parameters:')

set_button = st.sidebar.button('Set', on_click = callback_set)
if st.session_state.button_set_clicked:
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

        st.session_state['X_RTFE'] = st.number_input('X [m]', min_value=0.0,
                                        max_value=10000000.0, value=30.0, step=0.01)
        st.session_state['Y_RTFE'] = st.number_input('Y [m]', min_value=0.0, max_value=10000000.0, value=35.0, step=0.1)
        st.session_state['Z_RTFE'] = st.number_input('Z [m]', min_value=-5000.0, max_value=10000000.0, value=10.0, step=0.1)
        st.session_state['Az_i_RTFE'] = st.number_input('Initial azimuth [°]', min_value=-180.0, max_value=180.0, value=0.0, step=0.1)
        st.session_state['Zen_i_RTFE'] = st.number_input('Initial zenith [°]', min_value=0.0, max_value=180.0, value=0.0, step=0.1)
        st.session_state['Az_constr'] = st.number_input('Azimuth constraint [°]', min_value=0.0, max_value=180.0, value=180.0, step=0.01)
        st.session_state['Zen_constr'] = st.number_input('Zenith constraint [°]', min_value=0.0, max_value=92.0, value=92.0, step=0.01)
        st.session_state['DL_constr'] = st.number_input('Dogleg constraint [°/10m]', min_value=-180.0, max_value=180.0, value=2.0, step=0.1)
        st.session_state['Step_L'] = st.number_input('Length of 1 step [m]', min_value=0.0, max_value=50.0, value=5.0, step=1.0)

        submit_button = st.form_submit_button(label='Submit')



#####################################################################################################################  

def callback_launch():
    st.session_state.button_clicked_data = False
    st.session_state.button_clicked = False
    st.session_state.button_set_clicked = False
    st.session_state.button_launch_clicked = True

st.sidebar.subheader('RTFE simulation:')
st.sidebar.button('Launch', on_click = callback_launch)
if st.session_state.button_launch_clicked:
    st.subheader('RTFE simulation')
    st.write(f'Selected parameters  \n '
             f"Initial coordinates: {st.session_state['X_RTFE'], st.session_state['Y_RTFE'], st.session_state['Z_RTFE']} \n"
             f'Initial azimuth: {st.session_state["Az_i_RTFE"]} \n'
             f'Initial zenith: {st.session_state["Zen_i_RTFE"]} \n'
             f'Zenith constraint: {st.session_state["Zen_constr"]} \n'
             f'Azimuth constraint: {st.session_state["Az_constr"]} \n'
             f'Dogleg constraint: {st.session_state["Az_constr"]} \n'
             f'Step length: {st.session_state["Step_L"]} \n'
             )

    if st.session_state['engine'] == 'Evolution algorithm':
        cube = upload_selected_file(st.session_state['file_path'])
        engine = DE_algo(cube)
        with st.spinner('Planning the trajectory..'):
            OFV, traj, df = engine.DE_planning(

                bounds=[(-20, st.session_state['Az_constr']),(-20, st.session_state['Az_constr']),(-20, st.session_state['Az_constr']),
                        (0, st.session_state['Zen_constr']),(0, st.session_state['Zen_constr']),(0, st.session_state['Zen_constr'])],
                length=st.session_state['Step_L'], angle_constraint=st.session_state['DL_constr'],
                init_incl=[st.session_state['Zen_i_RTFE'], st.session_state['Zen_i_RTFE']],
                init_azi=[st.session_state['Az_i_RTFE'], st.session_state['Az_i_RTFE']],
               init_pos=[st.session_state['X_RTFE'], st.session_state['Y_RTFE'], st.session_state['Z_RTFE']])

        st.success(f'Objective function value : {round(float(OFV),2)}')

        st.session_state['ready_engine'] = engine
        st.session_state['trajectory'] = traj
        st.session_state['OFV'] = round(float(OFV), 2)
        st.session_state['trajectory_table'] = df

        st.write('')

        st.subheader('Trajectory 3D view')
        fig = traj_plot(traj[0],traj[1],traj[2])
        st.pyplot(fig)



        st.write('Later')

        st.subheader('Plan view')
        cube = upload_selected_file(st.session_state['file_path'])

        fig = plot_results(cube, st.session_state['trajectory'][0], st.session_state['trajectory'][1],
                           st.session_state['trajectory'][2])
        st.pyplot(fig)
        # st.subheader('Section view')
        # image_path = str(Path.joinpath(Path(os.getcwd()).parent, Path("figs") / "2D_ZY.PNG"))
        # MW_image = Image.open(image_path)
        # st.image(MW_image)

        st.subheader('Drilling trajectory table')
        st.dataframe(st.session_state['trajectory_table'])

        st.subheader('OFV values')
        Plan_tr = 468

        st.write('Planned trajectory OFV:', Plan_tr)
        st.write('Corrected trajectory OFV:', st.session_state['OFV'])


    #st.write('Assign RTFE parameters into algorithm')
    
else:
    st.sidebar.write('')
    
#####################################################################################################################     


