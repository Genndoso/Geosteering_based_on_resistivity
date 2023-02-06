import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import pickle
from scipy.interpolate import interpn
import pandas as pd
#
#
# class DE_algo:
#     def __init__(self, cube_3d):
#         self.cube_3d = cube_3d
#
#         x = np.linspace(0, cube_3d.shape[0] - 1, cube_3d.shape[0])
#         y = np.linspace(0, cube_3d.shape[1] - 1, cube_3d.shape[1])
#         z = np.linspace(0, cube_3d.shape[2] - 1, cube_3d.shape[2])
#         self.points = (x, y, z)
#
#     def get_vec(self, inc, azi, length, nev=False, deg=True):
#         """
#         Convert inc and azi into a vector.
#         Params:
#             inc: array of n floats
#                 Inclination relative to the z-axis (up)
#             azi: array of n floats
#                 Azimuth relative to the y-axis
#             r: float or array of n floats
#                 Scalar to return a scaled vector
#         Returns:
#             An (n,3) array of vectors
#         """
#         if deg:
#             inc_rad, azi_rad = np.radians(np.array([inc, azi]))
#         else:
#             inc_rad = inc
#             azi_rad = azi
#         y = length * np.sin(inc_rad) * np.cos(azi_rad)
#         x = length * np.sin(inc_rad) * np.sin(azi_rad)
#         z = length * np.cos(inc_rad)
#
#         #     if nev:
#         #         vec = np.array([y, x, z]).T
#         #     else:
#         #         vec = np.array([x, y, z]).T
#         return np.stack([x, y, z])
#
#     def obj(self, angles, *state):
#         OFV = 0
#         penalty = 0
#         # vec_diff = self.get_vec(angles[0], angles[1], state[-1])
#         azi_l = [state[2][-1]]
#         incl_l = [state[1][-1]]
#         state_new = [state[0][0], state[0][1], state[0][2]]
#
#         if state_new[2] + 3 * state[4] >= self.cube_3d.shape[2]:
#             vec_diff = self.get_vec(angles[0], angles[3], state[-1])
#             state_new = [state_new[0] + vec_diff[0], state_new[1] + vec_diff[1], state_new[2] + vec_diff[2]]
#             azi_l.append(angles[0])
#             incl_l.append(angles[3])
#             dls_val = np.linalg.norm(angles[0] - azi_l[-1]) + np.linalg.norm(angles[3] - incl_l[-1])
#             if dls_val >= state[3] * state[4]:
#                 penalty += dls_val * 0.5
#
#             # contraint for high length action
#             length_constraint = np.linalg.norm(vec_diff)
#
#             length_rew = 1 - (self.cube_3d.shape[2] - state_new[2] / self.cube_3d.shape[2])
#
#             # objective function
#             OFV += interpn(self.points, self.cube_3d, state_new, method='nearest') / length_constraint - penalty
#             OFV += length_rew * 10
#
#         else:
#             for i in range(3):
#                 vec_diff = self.get_vec(angles[i], angles[3 + i], state[-1])
#
#                 state_new = [state_new[0] + vec_diff[0], state_new[1] + vec_diff[1], state_new[2] + vec_diff[2]]
#
#                 azi_l.append(angles[i])
#                 incl_l.append(angles[3 + i])
#                 # dogleg severity constraint
#                 dls_val = np.linalg.norm(angles[i] - azi_l[-1]) + np.linalg.norm(angles[3 + i] - incl_l[-1])
#                 if dls_val >= state[3] * state[4]:
#                     penalty += dls_val * 0.5
#
#                 # contraint for high length action
#                 length_constraint = np.linalg.norm(vec_diff)
#
#                 length_rew = 1 - (self.cube_3d.shape[2] - state_new[2] / self.cube_3d.shape[2])
#
#                 # objective function
#                 OFV += interpn(self.points, self.cube_3d, state_new, method='nearest') / length_constraint - penalty
#                 OFV += length_rew * 10
#         return -OFV
#
#     def DE_planning(self, pop_size=100, num_iters=1000, F=0.7, cr=0.7,
#                     bounds=[(0, 180), (0, 180), (0, 180), (0, 92), (0, 92), (0, 92)],
#                     length=10, angle_constraint=2,
#                     init_incl=[0, 0], init_azi=[10, 10],
#                     init_pos=[120, 170, 40]):
#         """
#         1 step differential evolution trajectory planning.
#         Params:
#             pop_size: int
#              population size
#             num_iters: int
#              define number of iterations
#             F: float (0,1)
#              scale factor for mutation
#             cr: flaot (0,1)
#              crossover rate for recombination
#             bounds: list of tuples
#              bound for searchable paramaters (in our case (azi, inclination))
#             angle_constraint: float
#              dogleg constraint per m
#             length: int
#              length of one step
#
#         """
#         OFV = 0
#         pos = init_pos
#         incl_l = init_incl
#         azi_l = init_azi
#         state = ([pos[0], pos[1], pos[2]], incl_l, azi_l, angle_constraint, length)
#         traj_x = [state[0][0]]
#         traj_y = [state[0][1]]
#         traj_z = [state[0][2]]
#         с = 0
#
#         while (traj_z[-1] <= self.cube_3d.shape[2] - 20) and (traj_y[-1] <= self.cube_3d.shape[1] - 20) and \
#                 (traj_x[-1] <= self.cube_3d.shape[0] - 20):
#             de_sol = differential_evolution(self.obj, bounds, args=(state), mutation=F, popsize=pop_size,
#                                             maxiter=num_iters, updating='deferred', disp=False).x
#             incl_l.append(de_sol[0])
#             azi_l.append(de_sol[3])
#             step = self.get_vec(incl_l[-1], azi_l[-1], length=length)
#
#             traj_x.append(state[0][0] + step[0])
#             traj_y.append(state[0][1] + step[1])
#             traj_z.append(state[0][2] + step[2])
#             print(с, traj_x[-1], traj_y[-1], traj_z[-1])
#             state[0][0] = state[0][0] + step[0]
#             state[0][1] = state[0][1] + step[1]
#             state[0][2] = state[0][2] + step[2]
#             state_new = [state[0][0], state[0][1], state[0][2]]
#
#             OFV += interpn(self.points, self.cube_3d, state_new, method='nearest')
#             с += 1
#         print(OFV)
#         return OFV, np.stack([traj_x, traj_y, traj_z])















class DE_algo:
    def __init__(self, cube_3d):
        new_cube = np.zeros(shape=(cube_3d.shape[0] + 20, cube_3d.shape[1] + 20, cube_3d.shape[2] + 20))
        new_cube[:cube_3d.shape[0], :cube_3d.shape[1], :cube_3d.shape[2]] = cube_3d
        self.cube_3d = new_cube
        self.old_cube = cube_3d
        x = np.linspace(0, new_cube.shape[0] - 1, new_cube.shape[0])
        y = np.linspace(0, new_cube.shape[1] - 1, new_cube.shape[1])
        z = np.linspace(0, new_cube.shape[2] - 1, new_cube.shape[2])
        self.points = (x, y, z)

    def get_vec(self, inc, azi, length, nev=False, deg=True):
        """
        Convert inc and azi into a vector.
        Params:
            inc: array of n floats
                Inclination relative to the z-axis (up)
            azi: array of n floats
                Azimuth relative to the y-axis
            r: float or array of n floats
                Scalar to return a scaled vector
        Returns:
            An (n,3) array of vectors
        """
        if deg:
            inc_rad, azi_rad = np.radians(np.array([inc, azi]))
        else:
            inc_rad = inc
            azi_rad = azi
        y = length * np.sin(inc_rad) * np.cos(azi_rad)
        x = length * np.sin(inc_rad) * np.sin(azi_rad)
        z = length * np.cos(inc_rad)

        #     if nev:
        #         vec = np.array([y, x, z]).T
        #     else:
        #         vec = np.array([x, y, z]).T
        return np.stack([x, y, z])

    def obj(self, angles, *state):
        penalty = 0
        vec_diff = self.get_vec(angles[0], angles[1], state[-1])
        #  vec_diff = get_vec(angles[0], 10)
        state_new = [state[0][0] + vec_diff[0], state[0][1] + vec_diff[1], state[0][2] + vec_diff[2]]

        # dogleg severity constraint
        dls_val = np.linalg.norm(angles[0] - state[1][-2]) + np.linalg.norm(angles[1] - state[2][-2])
        if dls_val >= state[3] * state[4]:
            penalty += dls_val * 0.5

        # contraint for high length action
        length_constraint = np.linalg.norm(vec_diff)

        # objective function
        OFV = interpn(self.points, self.cube_3d, state_new, method='nearest') / length_constraint - penalty

        #length_rew = 1 - (self.cube_3d.shape[2] - state_new[2] / self.cube_3d.shape[2])
        OFV = OFV #+ length_rew*1.5

        return -OFV



    def DE_planning(self, pop_size=100, num_iters=1000, F=0.7, cr=0.7, bounds=[(0, 180), (0, 92)],
                    length=10, angle_constraint=0.1,
                    init_incl=[0, 0], init_azi=[10, 10],
                    init_pos=[30, 30, 150]):
        """
        1 step differential evolution trajectory planning.
        Params:
            pop_size: int
             population size
            num_iters: int
             define number of iterations
            F: float (0,1)
             scale factor for mutation
            cr: flaot (0,1)
             crossover rate for recombination
            bounds: list of tuples
             bound for searchable paramaters (in our case (azi, inclination))
            angle_constraint: float
             dogleg constraint per m
            length: int
             length of one step

        """
        OFV = 0
        pos = init_pos
        incl_l = init_incl
        azi_l = init_azi
        state = ([pos[0], pos[1], pos[2]], incl_l, azi_l, angle_constraint, length)
        traj_x = [state[0][0]]
        traj_y = [state[0][1]]
        traj_z = [state[0][2]]
        c = 0
        dic = {'Step': [],
               'X': [],
               'Y': [],
               'Z': []}
        while (traj_z[-1] <= self.old_cube.shape[2] - 2.1*length) and (traj_y[-1] <= self.old_cube.shape[1] - 2.1*length) and \
                (traj_x[-1] <= self.old_cube.shape[0] - 2.1*length):

            de_sol = differential_evolution(self.obj, bounds, args=(state), mutation = F, popsize = pop_size, \
                                            maxiter = num_iters, updating='deferred', disp=False).x
            incl_l.append(de_sol[0])
            azi_l.append(de_sol[1])
            step = self.get_vec(incl_l[-1], azi_l[-1], length=length)

            traj_x.append(state[0][0] + step[0])
            traj_y.append(state[0][1] + step[1])
            traj_z.append(state[0][2] + step[2])

            state[0][0] = state[0][0] + step[0]
            state[0][1] = state[0][1] + step[1]
            state[0][2] = state[0][2] + step[2]
            state_new = [state[0][0], state[0][1], state[0][2]]

            print(f'Step - {c}, x : {round(traj_x[-1],3)}, y: {round(traj_y[-1],3)}, z: {round(traj_z[-1],3)}')
            dic['Step'].append(c)
            dic['X'].append(round(traj_x[-1],3))
            dic['Y'].append(round(traj_y[-1], 3))
            dic['Z'].append(round(traj_z[-1], 3))
            OFV += interpn(self.points, self.cube_3d, state_new, method='nearest')
            c += 1
        print(OFV)
        df = pd.DataFrame(dic)
        return OFV, np.stack([traj_x, traj_y, traj_z]), df


def plot_results(volume_cube, traj_x, traj_y, traj_z):
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

    fig, ax = plt.subplots(3, 1, figsize=(15, 10))
    property_along_y_arr = np.array(property_along_y)
    property_along_x_arr = np.array(property_along_x)
    property_along_z_arr = np.array(property_along_z)

    #     #ax.imshow(property_along_z_arr.T[::-1])
    #     if along_axis == 'y':
    #         mean_y = round(np.array(traj_y).mean())
    #         p_map = plt.imshow(cube_3d[:,mean_y,:])
    #       #  p_map = ax.imshow(property_along_y_arr, aspect='auto')
    #         ax.plot(y_t, color = 'r', linewidth = 3 )
    #     elif along_axis == 'z':
    #         mean_x = round(np.array(traj_x).mean())
    #         p_map = plt.imshow(cube_3d[mean_x,:,:])
    #        # p_map =  ax.imshow(property_along_z_arr.T, aspect='auto')
    #         ax.plot(z_t,y_t, color = 'r', linewidth = 3 )
    #     else:
    #         mean_z = round(np.array(traj_z).mean())
    #         p_map = plt.imshow(cube_3d[:,:,mean_z])
    #       #  p_map = ax.imshow(property_along_x_arr, aspect='auto')
    #         ax.plot(x_t, color = 'r', linewidth = 3 )
    #     plt.colorbar(p_map)
    mean_x = round(np.array(traj_x).mean())
    ax[0].plot(z_t, y_t, color='r', linewidth=3)
    ax[0].set_title('XZ trajectory projection')
    ax[0].imshow(volume_cube[mean_x, :, :])

    mean_y = round(np.array(traj_y).mean())
    ax[1].plot(z_t, x_t, color='r', linewidth=3)
    ax[1].set_title('YZ trajectory projection')
    ax[1].imshow(volume_cube[:, mean_y, :])

    ax[2].set_title('XY trajectory projection')
    mean_z = round(np.array(traj_z).mean())
    ax[2].imshow(volume_cube[:, :, mean_z])
    ax[2].plot(x_t, y_t, color='r', linewidth=3)
    return fig
