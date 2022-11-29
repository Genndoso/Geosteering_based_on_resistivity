import numpy as np
from scipy.interpolate import interpn

import itertools
volume_cut = np.random.uniform(0, 10, size=(301, 514, 278))


class greedy_geosteering_polar:
    def __init__(self, map_3d, length=12, angle_constraint_per_m=0.1, steps_ahead=3, start_point=[30, 30, 20],
                 init_inclination=86, init_azimut=0,
                 step_incl=0.5, step_azi=0.5,
                 min_azimut=0, max_azimut=270,
                 min_zenith=70, max_zenith=92):
        self.map_3d = map_3d

        self.length = length
        self.min_zenith = min_zenith
        self.max_zenith = max_zenith
        self.min_azimut = min_azimut
        self.max_azimut = max_azimut
        self.angle_constraint = angle_constraint_per_m
        self.steps_ahead = steps_ahead

        self.traj_x = [start_point[0]]
        self.traj_y = [start_point[1]]
        self.traj_z = [start_point[2]]

        self.incl_l = [init_inclination]
        self.azi_l = [init_azimut]

        self.step_incl = step_incl
        self.step_azimut = step_azi

        self.x = np.linspace(0, self.map_3d.shape[0] - 1, self.map_3d.shape[0])
        self.y = np.linspace(0, self.map_3d.shape[1] - 1, self.map_3d.shape[1])
        self.z = np.linspace(0, self.map_3d.shape[2] - 1, self.map_3d.shape[2])
        self.points = (self.x, self.y, self.z)

    def get_vec(self, inc, azi, nev=False, deg=True):

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
        y = self.length * np.sin(inc_rad) * np.cos(azi_rad)
        x = self.length * np.sin(inc_rad) * np.sin(azi_rad)
        z = self.length * np.cos(inc_rad)

        #     if nev:
        #         vec = np.array([y, x, z]).T
        #     else:
        #         vec = np.array([x, y, z]).T
        return np.stack([x, y, z])

    def get_dogleg(self, inc1, azi1, inc2, azi2):

        inc1 = np.radians(inc1)
        inc2 = np.radians(inc2)
        azi1 = np.radians(azi1)
        azi2 = np.radians(azi2)
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

    def get_best_candidate(self, current_point):
        OFV = 0
        OFV_best = 0
        all_candidates = []
        break_all = False
        items = [[-self.step_incl, self.step_azimut], [-self.step_incl, -self.step_azimut], [-self.step_incl, 0],
                 [0, 0],
                 [0, -self.step_azimut], [0, self.step_azimut],
                 [self.step_incl, self.step_azimut], [self.step_incl, 0], [self.step_incl, -self.step_azimut]]

        for item in itertools.product(items, repeat=self.steps_ahead):
            all_candidates.append(list(item))

        best_candidate = all_candidates[0]
        cand_point = current_point
        OFV_best = 0

        incl = self.incl_l[-1]
        azi = self.azi_l[-1]
        #   cand_point = [self.traj_x[-1],traj_y[-1], traj_z[-1]]
        for l in range(0, len(all_candidates)):
            OFV = 0
            cand_point = current_point
            for v in range(0, len(all_candidates[1])):
                incl += all_candidates[l][v][0]
                azi += all_candidates[l][v][1]
                incl_arr = np.array(incl)
                azi_arr = np.array(azi)

                vec = self.get_vec(incl_arr, azi_arr, self.length)
                cand_point = [cand_point[0] + vec[0], cand_point[1] + vec[1], cand_point[2] + vec[2]]
                # interpolate between points
                try:
                    if l == 0:
                        OFV_best += interpn(self.points, self.map_3d, cand_point, method='nearest') / np.linalg.norm(
                            vec)
                    else:
                        OFV += interpn(self.points, self.map_3d, cand_point, method='nearest') / np.linalg.norm(vec)
                except:
                    break_all = True

        if OFV > OFV_best:
            best_candidate = all_candidates[l]
            OFV_best = OFV

        return best_candidate, break_all

    def get_next_step(self, z, step_back=1):
        k = 0
        next_point = [self.traj_x[-1], self.traj_y[-1], self.traj_z[-1]]
        traj_x_array, traj_y_array, traj_z_array = np.stack([self.traj_x, self.traj_y, self.traj_z])
        best_candidate, break_all = self.get_best_candidate(next_point)
        OFV = 0
        break_al = False
        if z < 2:
            dogleg = 0
            next_incl_diff = best_candidate[0][0]
            next_azi_diff = best_candidate[0][1]
        else:
            incl2 = self.incl_l[-1]
            azi2 = self.azi_l[-1]
            incl1 = self.incl_l[-1 - step_back]
            azi1 = self.azi_l[-1 - step_back]
            #     dogleg = self.get_dogleg(incl1, azi1, incl2, azi2)

            # limit zenith angle
            if incl2 >= self.max_zenith:
                next_incl_diff = - self.step_incl
            elif incl2 <= self.min_zenith:
                next_incl_diff = + self.step_incl
            else:
                next_incl_diff = best_candidate[0][0]

            # limith azimut angle
            if azi2 >= self.max_azimut:
                next_azi_diff = -self.step_azimut
            elif azi2 <= self.min_azimut:
                next_azi_diff = self.step_azimut
            else:
                next_azi_diff = best_candidate[0][1]

            # constraint dogleg severity
            #  if np.degrees(dogleg) >= self.angle_constraint:

            dls_val = np.linalg.norm(incl1 - incl2) + np.linalg.norm(azi1 - azi2)
            if dls_val >= self.angle_constraint * self.length:
                next_incl_diff = 0
                next_azi_diff = 0

        #  next_azi_diff = best_candidate[0][1]
        #  next_incl_diff = best_candidate[0][1]

        k = 0
        # print(next_azi_diff, next_incl_diff)
        if (next_point[0] >= self.map_3d.shape[0] - 30) or (next_point[0] <= 10):
            next_step = [0, 1, 0]
            k = 1
            break_al = True
        if (next_point[1] >= self.map_3d.shape[1] - 30) or (next_point[1] <= 10):
            next_step = [0, 1, 0]
            break_al = True
        if (next_point[2] >= self.map_3d.shape[2] - 30) or (next_point[2] <= 10):
            next_step = [0, 1, 0]
            break_al = True

        if k == 1:
            vec_diff = next_step
        else:
            self.incl_l.append(self.incl_l[-1] + next_incl_diff)
            self.azi_l.append(self.azi_l[-1] + next_azi_diff)
            vec_diff = self.get_vec(self.incl_l[-1], self.azi_l[-1], self.length)

        self.traj_x.append(self.traj_x[-1] + vec_diff[0])
        self.traj_y.append(self.traj_y[-1] + vec_diff[1])
        self.traj_z.append(self.traj_z[-1] + vec_diff[2])
        point_new = np.array([self.traj_x[-1], self.traj_y[-1], self.traj_z[-1]])
        point_last = np.array([self.traj_x[-2], self.traj_y[-2], self.traj_z[-2]])

        OFV += interpn(self.points, self.map_3d, [self.traj_x[-1], self.traj_y[-1], self.traj_z[-1]], method='nearest') \
               / np.linalg.norm(point_new - point_last)
        # print(self.incl_l[-1],self.azi_l[-1], point_new, best_candidate)
        print(self.incl_l[-1], self.azi_l[-1], point_new)
        return OFV, break_al

    def traj_planning(self, step_back=1):
        OFV = 0
        z = 1

        while self.traj_y[-1] <= self.map_3d.shape[1] - 1:
            OFV_p, break_al = self.get_next_step(z=z, step_back=step_back)

            OFV += OFV_p

            if break_al:
                break

            z += 1
        print('OFV =', np.round(OFV, 2))
        print(z)
        return np.stack([self.traj_x, self.traj_y, self.traj_z])

    def reset(self, start_point, init_inclination, init_azimut):

        self.traj_x = [start_point[0]]
        self.traj_y = [start_point[1]]
        self.traj_z = [start_point[2]]

        self.incl_l = [init_inclination]
        self.azi_l = [init_azimut]






