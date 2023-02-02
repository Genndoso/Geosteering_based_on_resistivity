import k3d
import numpy as np
import streamlit as st
import pyvista as pv
from stpyvista import stpyvista
pv.global_theme.show_scalar_bar = False
import matplotlib.pyplot as plt

def visualize_cube(volume_cut, opacity):
    grid = pv.UniformGrid()

    # Set the grid dimensions: shape + 1 because we want to inject our values on
    #   the CELL data
    grid.dimensions = np.array(volume_cut.shape) + 1

    # Edit the spatial reference
    grid.origin = (80,80, 100) # The bottom left corner of the data set
    grid.spacing = (1, 1, 1)  # These are the cell sizes along each axis

    # Add the data values to the cell data
    grid.cell_data["values"] = volume_cut.flatten(order="F")  # Flatten the array!

    ## Initialize a plotter object
    plotter = pv.Plotter(window_size=[400, 400])

    actor = plotter.add_mesh(grid, label='grid', opacity=opacity, cmap='gist_rainbow_r')

   # actor.rotate_y(-90)
    # plotter.add_legend()

    ## Send to streamlit
    stpyvista(plotter, key="pv_cube")


def vis_2d(file):
    x_y_z_vis = st.selectbox("Choose a plane for plotting:", ["X-Y",
                                                              "Y-Z",
                                                              "X-Z"])

    column1, column2 = st.columns([1, 4])

    if x_y_z_vis == "X-Y":
        # Add slider to column 1
        slider = column1.slider("Slice # along X-Y plane", min_value=0, max_value=file.shape[2], value=1)
        # Add plot to column 2
        fig = plt.imshow(file[:, :, slider])
        st.pyplot(fig)

    elif x_y_z_vis == "Y-Z":
        # Add slider to column 1
        slider = column1.slider("Slice # along Y-Z plane", min_value=0, max_value=file.shape[0], value=1)
        # Add plot to column 2
        fig = plt.imshow(file[slider, :,:])
        st.pyplot(fig)

    elif x_y_z_vis == "X-Z":
        # Add slider to column 1
        slider = column1.slider("Slice # along Y-Z plane", min_value=0, max_value=file.shape[1], value=1)
        # Add plot to column 2
        fig = plt.imshow(file[slider, :, :])
        st.pyplot(fig)


