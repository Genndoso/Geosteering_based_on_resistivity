import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import itertools
from scipy.signal import savgol_filter
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter, laplace
from scipy import ndimage
from scipy import interpolate
from mpl_toolkits import mplot3d
from PIL import Image
import IPython
import pickle
# from vtk.util import numpy_support
# import vtk
import pyvista as pv
import plotly.graph_objects as go

items = [[-1,1], [-1, -1], [-1,0], [0,0], [0, -1], [0,1], [1,1], [1,0], [1,-1]]


class greedy_geosteering_advanced:
    def __init__(self, map_3d, items, step_z=1, angle_constraint=3, steps_ahead=3, min_azimut=0, max_azimut=360,
                 min_zenith=80, max_zenith=91):

        self.map_3d = map_3d
        self.items = items
        self.step_z = step_z
        self.min_zenith = min_zenith
        self.max_zenith = max_zenith
        self.min_azimut = min_azimut
        self.max_azimut = max_azimut
        self.angle_constraint = angle_constraint
        self.steps_ahead = steps_ahead

    def get_dogleg(self, inc1, azi1, inc2, azi2):
        dogleg = (
                2 * np.arcsin(
            (
                    np.sin((inc2 - inc1) / 2) ** 2
                    + np.sin(inc1) * np.sin(inc2)
                    * np.sin((azi2 - azi1) / 2) ** 2
            ) ** 0.5
        )
        )
        return dogleg

    def _get_angles(self, traj_x, traj_y, traj_z):
        xz = traj_x ** 2 + traj_z ** 2
        inc = np.arctan2(np.sqrt(xz), traj_y)  # for elevation angle defined from y-axis down
        azi = (np.arctan2(traj_x, traj_z) + (2 * np.pi)) % (2 * np.pi)

        return np.stack((inc, azi), axis=1)

    def get_best_candidate(self, current_point):
        all_candidates = []

        for item in itertools.product(items, repeat=self.steps_ahead):
            all_candidates.append(list(item))

        best_candidate = all_candidates[0]
        cand_point = current_point
        OFV_best = 0

        if cand_point[0] < (self.map_3d.shape[0] - self.steps_ahead) and \
                (cand_point[1] < self.map_3d.shape[1] - self.steps_ahead) and (
                cand_point[2] < self.map_3d.shape[2] - self.step_z - self.steps_ahead):
            for l in range(0, len(all_candidates)):
                #   obtain OFV of all possible candidates
                OFV = 0
                cand_point = current_point
                for v in range(0, len(all_candidates[1])):

                    cand_point = [cand_point[0] + all_candidates[l][v][0],
                                  cand_point[1] + all_candidates[l][v][1], cand_point[2] + 1]
                    if l == 0:
                        OFV_best += self.map_3d[cand_point[0], cand_point[1], cand_point[2]]
                    else:
                        OFV += self.map_3d[cand_point[0], cand_point[1], cand_point[2]]

                if OFV > OFV_best:
                    best_candidate = all_candidates[l]
                    OFV_best = OFV
        return best_candidate

    def get_next_step(self, traj_x, traj_y, traj_z, z, dx=0, dy=0, step_back=1):
        k = 0
        next_point = [traj_x[-1], traj_y[-1], traj_z[-1]]

        traj_x_array, traj_y_array, traj_z_array = np.stack([traj_x, traj_y, traj_z])
        angles = self._get_angles(traj_x_array, traj_y_array, traj_z_array)

        best_candidate = self.get_best_candidate(next_point)
        #  Calculate DogLeg
        if z >= step_back:
            incl2, az2 = angles[-step_back - 1]
        elif z >= 1 and z < step_back:
            incl2, az2 = angles[-1 - 1]

        incl1, az1 = angles[-1]

        if z > 1:
            dogleg = self.get_dogleg(incl1, az1, incl2, az2)
            print(z, np.degrees(incl2), np.degrees(dogleg), (np.degrees(az1)))

            if np.degrees(dogleg) >= self.angle_constraint:
                next_step = [dx, dy, self.step_z]
                k = 1
            if (np.degrees(incl2) <= 80) or (np.degrees(incl2) >= 95):
                next_step = [0, 0, self.step_z]
                k = 1

        if k != 1:

            next_step = [best_candidate[0][0], best_candidate[0][1], self.step_z]

        # upper boundary x,y constraint
        if next_point[0] >= self.map_3d.shape[0] - self.step_z:
            if next_point[1] >= self.map_3d.shape[1] - self.step_z:
                next_step = [-self.step_x, - self.step_x, self.step_z]
            elif next_point[1] <= self.step_z:
                next_step = [-self.step_x, 0, self.step_z]
            else:
                next_step = [0, 0, self.step_z]

        # lower boundary x,x constraint
        if next_point[0] <= self.step_z:
            if next_point[1] >= self.map_3d.shape[1] - self.step_z:
                next_step = [0, - self.step_x, self.step_z]
            elif next_point[1] <= self.step_z:
                next_step = [0, 0, self.step_z]
            else:
                next_step = [0, 0, self.step_z]

        # upper boundary y,x constraint
        if next_point[1] >= self.map_3d.shape[1] - self.step_z:
            if next_point[0] >= self.map_3d.shape[0] - self.step_z:
                next_step = [-self.step_x, - self.step_x, self.step_z]
            else:
                next_step = [0, - self.step_x, self.step_z]

            # lower boundary y,x constraint
        if next_point[1] <= self.step_z:
            if next_point[0] >= self.map_3d.shape[0] - self.step_z:
                next_step = [-self.step_x, 0, self.step_z]
            else:
                next_step = [0, 0, self.step_z]

        return next_step

    def traj_planning(self, start_point, step_back=10):
        OFV = 0
        next_point = start_point
        traj_x = [start_point[0]]
        traj_y = [start_point[1]]
        traj_z = [start_point[2]]
        greedy_simple = False
        dx = 0
        dy = 0

        for z in range(0, self.map_3d.shape[2] - self.steps_ahead - start_point[2] - self.step_z, self.step_z):
            next_step = self.get_next_step(traj_x, traj_y, traj_z, dx=0, dy=0, z=z, step_back=step_back)

            next_point = [next_point[0] + next_step[0], next_point[1] + next_step[1], next_point[2] + next_step[2]]

            if z > 0:
                dx = next_point[0] - next_point_prev[0]
                dy = next_point[1] - next_point_prev[1]
            OFV += self.map_3d[next_point[0], next_point[1], next_point[2]]
            next_point_prev = next_point
            traj_x.append(next_point[0])
            traj_y.append(next_point[1])
            traj_z.append(next_point[2])

        print('OFV =', np.round(OFV, 2))
        return np.stack([traj_x, traj_y, traj_z])

    @staticmethod
    def trajectory_visualization(traj_x, traj_y, traj_z, color='red', width=7):
        fig = go.Figure()

        fig.add_trace(
            go.Scatter3d(
                x=traj_x,
                y=traj_z,
                z=traj_y,
                mode='lines',
                line=dict(
                    color=color,
                    width=width
                ),
                name='survey_interpolated'
            ),
        )
        fig.update_layout(scene=dict(
            #  xaxis_title='X AXIS TITLE',
            yaxis_title='Drilling direction',
            zaxis_title='True vertical depth (TVD)'),
            width=700,
            margin=dict(r=20, b=10, l=10, t=10))

        fig.update_scenes(zaxis_autorange="reversed")
        fig.show()






class Greedy_3d:
    def __init__(self, map_3d, n_steps_ahead, angle_constraint, step_z=1):
        self.map_3d = map_3d
        self.n_steps = n_steps_ahead
        self.angle_constraint = angle_constraint
        self.step_z = step_z
        self.step_x = 1
        self.step_y = 1

    def get_dogleg(self, inc1, azi1, inc2, azi2):
        dogleg = (
                2 * np.arcsin(
            (
                    np.sin((inc2 - inc1) / 2) ** 2
                    + np.sin(inc1) * np.sin(inc2)
                    * np.sin((azi2 - azi1) / 2) ** 2
            ) ** 0.5
        )
        )

        return dogleg

    def _get_angles(self, traj_x, traj_y, traj_z):
        xy = traj_x ** 2 + traj_y ** 2
        inc = np.arctan2(np.sqrt(xy), traj_z)  # for elevation angle defined from Z-axis down
        azi = (np.arctan2(traj_x, traj_y) + (2 * np.pi)) % (2 * np.pi)

        return np.stack((inc, azi), axis=1)

    def traj_planning(self, start_point, items, step_back=1):
        # traj = np.zeros_like(self.map_3d)

        # traj[start_point[0], start_point[1], start_point[2]] = 1
        OFV = 0
        k = 0
        next_point = start_point
        all_candidates = []
        traj_x = [start_point[0]]
        traj_y = [start_point[1]]
        traj_z = [start_point[2]]
        greedy_simple = False

        for item in itertools.product(items, repeat=self.n_steps):
            all_candidates.append(list(item))

        for z in range(0, self.map_3d.shape[2] - self.n_steps - start_point[2]):
            best_candidate = all_candidates[0]
            cand_point = next_point
            OFV_best = 0

            if cand_point[0] < (self.map_3d.shape[0] - self.n_steps) and \
                    (cand_point[1] < self.map_3d.shape[1] - self.n_steps):
                for l in range(0, len(all_candidates)):
                    #   obtain OFV of all possible candidates
                    OFV = 0
                    cand_point = next_point
                    for v in range(0, len(all_candidates[1])):

                        cand_point = [cand_point[0] + all_candidates[l][v][0],
                                      cand_point[1] + all_candidates[l][v][1], cand_point[2] + 1]
                        if l == 0:

                            OFV_best += self.map_3d[cand_point[0], cand_point[1], cand_point[2]]
                        else:
                            OFV += self.map_3d[cand_point[0], cand_point[1], cand_point[2]]

                    if OFV > OFV_best:
                        best_candidate = all_candidates[l]
                        OFV_best = OFV
            # angle constraint based on dogleg severity
            # прирост углов между локальнымил линейными приближениями, нормированные по приросту длины

            # linear approximation

            #  Calculate DogLeg
            if z >= step_back + 2:
                #                 az1 = np.arccos((traj_x[-1] - traj_x[-2])/ self.step_z)
                #                 az2 = np.arccos((traj_x[-1 - step_back] - traj_x[-2 - step_back])/ self.step_z)
                #                 incl1 = np.arccos((traj_y[-1] - traj_y[-2])/ self.step_z)
                #                 incl2 = np.arccos((traj_y[-1 - step_back] - traj_y[-2 - step_back])/ self.step_z)

                #                 d_incl = (incl1 - incl2)
                #                 d_az = (az1 - az2)

                traj_x_array, traj_y_array, traj_z_array = np.stack([traj_x, traj_y, traj_z])

                angles = self._get_angles(traj_x_array, traj_y_array, traj_z_array)

                incl1, az1 = angles[-1]

                incl2, az2 = angles[-step_back - 1]

                print('dincl1', np.degrees(incl2-incl1), '\n daz', np.degrees(az2-az1))
                dogleg = self.get_dogleg(incl1, az1, incl2, az2)

                if np.degrees(dogleg) >= self.angle_constraint:
                    next_step = [dx, dy, self.step_z]
                    k = 1
            #     traj_y_targ = self.step_z*np.cos(self.angle_constraint+incl2)+traj_y[-2] #corrected y_target based on the restriction of dogleg
            # also possible is to do search on a half-sphere via the scipy optimize framework, but it needs to be restructured for your
            # problem (I had a distinct target endpoint)
            # here you could define the cost function based on your oil saturation
            # and regularise with softplus (ln(1+e^kx); smooth approximation of ReLU) of dogleg shifted to the maximum tolerance point

            if not greedy_simple and k != 1:
                next_step = [best_candidate[0][0], best_candidate[0][1], self.step_z]

            # upper boundary x,y constraint
            if next_point[0] >= self.map_3d.shape[0] - self.step_z:
                if next_point[1] >= self.map_3d.shape[1] - self.step_z:
                    next_step = [-self.step_x, - self.step_x, self.step_z]
                elif next_point[1] <= self.step_z:
                    next_step = [-self.step_x, 0, self.step_z]
                else:
                    next_step = [0, 0, self.step_z]

                    # lower boundary x,x constraint
            if next_point[0] <= self.step_z:
                if next_point[1] >= self.map_3d.shape[1] - self.step_z:
                    next_step = [0, - self.step_x, self.step_z]
                elif next_point[1] <= self.step_z:
                    next_step = [0, 0, self.step_z]
                else:
                    next_step = [0, 0, self.step_z]

                # upper boundary y,x constraint
            if next_point[1] >= self.map_3d.shape[1] - self.step_z:
                if next_point[0] >= self.map_3d.shape[0] - self.step_z:
                    next_step = [-self.step_x, - self.step_x, self.step_z]
                else:
                    next_step = [0, - self.step_x, self.step_z]

                # lower boundary y,x constraint
            if next_point[1] <= self.step_z:
                if next_point[0] >= self.map_3d.shape[0] - self.step_z:
                    next_step = [-self.step_x, 0, self.step_z]
                else:
                    # next_point[0] <= self.step_z:
                    next_step = [0, 0, self.step_z]

            # print(next_point, next_step)

            next_point = [next_point[0] + next_step[0], next_point[1] + next_step[1], next_point[2] + next_step[2]]

            if z > 1:
                dx = next_point[0] - next_point_prev[0]
                dy = next_point[1] - next_point_prev[1]

            OFV += self.map_3d[next_point[0], next_point[1], next_point[2]]
            next_point_prev = next_point
            traj_x.append(next_point[0])
            traj_y.append(next_point[1])
            traj_z.append(next_point[2])

        print('OFV =', np.round(OFV, 2))
        return np.stack([traj_x, traj_y, traj_z])

    @staticmethod
    def trajectory_visualization(traj_x, traj_y, traj_z, color = 'red', width = 7):
        fig = go.Figure()

        fig.add_trace(
            go.Scatter3d(
                x=traj_x,
                y=traj_z,
                z=traj_y,
                mode='lines',
                line=dict(
                    color= color,
                    width= width
                ),
                name='survey_interpolated'
            ),
        )
        fig.update_layout(scene=dict(
            #  xaxis_title='X AXIS TITLE',
            yaxis_title='Drilling direction',
            zaxis_title='True vertical depth (TVD)'),
            width=700,
            margin=dict(r=20, b=10, l=10, t=10))

        fig.update_scenes(zaxis_autorange="reversed")
        fig.show()


