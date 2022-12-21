from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()


def plot_results(volume_cube, traj_x, traj_y, traj_z, along_axis='y', index = 22):
    x = []
    z = []
    y = []
    property_along_y = []
    property_along_x = []
    property_along_z = []
    for i in range(0, len(traj_x)):
        x.append(traj_x[i])
        z.append(traj_z[i])
        y.append(traj_y[i])
        property_along_y.append(volume_cube[round(traj_x[i]), :, round(traj_z[i])].T)
        property_along_x.append(volume_cube[:, round(traj_y[i]), round(traj_z[i])].T)
        property_along_z.append(volume_cube[round(traj_x[i]), round(traj_y[i]), :].T)

    x_t = np.array(x)
    z_t = np.array(z)
    y_t = np.array(y)

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    property_along_y_arr = np.array(property_along_y)
    property_along_x_arr = np.array(property_along_x)
    property_along_z_arr = np.array(property_along_z)
    #  ax.plot(z_t, color = 'r', linewidth = 3 )
    ax.set_title('XZ trajectory projection')

    # ax.imshow(property_along_z_arr.T[::-1])
    if along_axis == 'y':
        p_map = plt.imshow(volume_cube[:, round(traj_y[index]), :])
        #  p_map = ax.imshow(property_along_y_arr, aspect='auto')
        ax.plot(y_t, color='r', linewidth=3)
    elif along_axis == 'z':
        p_map = plt.imshow(volume_cube[round(traj_x[index]), :, :])
        # p_map =  ax.imshow(property_along_z_arr.T, aspect='auto')
        ax.plot(z_t, y_t, color='r', linewidth=3)
    else:
        p_map = plt.imshow(volume_cube[:, :, round(traj_z[index])])
        #  p_map = ax.imshow(property_along_x_arr, aspect='auto')
        ax.plot(x_t, color='r', linewidth=3)
    plt.colorbar(p_map)

