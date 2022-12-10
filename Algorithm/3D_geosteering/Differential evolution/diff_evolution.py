import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import pickle
from scipy.interpolate import interpn


class DE_algo:
    def __init__(self, cube_3d):
        self.cube_3d = cube_3d

        x = np.linspace(0, cube_3d.shape[0] - 1, cube_3d.shape[0])
        y = np.linspace(0, cube_3d.shape[1] - 1, cube_3d.shape[1])
        z = np.linspace(0, cube_3d.shape[2] - 1, cube_3d.shape[2])
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
        return -OFV

    def DE_planning(self, pop_size=100, num_iters=1000, F=0.7, cr=0.7, bounds=[(0, 180), (0, 92)],
                    length=10, angle_constraint=0.1,
                    init_incl=[0, 0], init_azi=[10, 10],
                    init_pos=[120, 170, 40]):
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
        с = 0

        while (traj_z[-1] <= self.cube_3d.shape[2] - 20) and (traj_y[-1] <= self.cube_3d.shape[1] - 20) and \
                (traj_x[-1] <= self.cube_3d.shape[0] - 20):
            de_sol = differential_evolution(self.obj, bounds, args=(state), mutation = F, popsize = pop_size, maxiter = num_iters, updating='deferred', disp=False).x
            incl_l.append(de_sol[0])
            azi_l.append(de_sol[1])
            step = self.get_vec(incl_l[-1], azi_l[-1], length=length)

            traj_x.append(state[0][0] + step[0])
            traj_y.append(state[0][1] + step[1])
            traj_z.append(state[0][2] + step[2])
            print(с, traj_x[-1], traj_y[-1], traj_z[-1])
            state[0][0] = state[0][0] + step[0]
            state[0][1] = state[0][1] + step[1]
            state[0][2] = state[0][2] + step[2]
            state_new = [state[0][0], state[0][1], state[0][2]]

            OFV += interpn(self.points, self.cube_3d, state_new, method='nearest')
            с += 1
        print(OFV)
        return OFV, np.stack([traj_x, traj_y, traj_z])