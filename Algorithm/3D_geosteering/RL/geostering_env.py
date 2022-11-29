import itertools
import torch
from torch import nn
import gym
from gym.spaces import Discrete, Box
from gym import Env
from scipy.interpolate import interpn
import os
import numpy as np
from scipy.interpolate import interpn

class Geosteering_env(Env):
    def __init__(self,  prod_map: np.array, start_point=[0, 20, 20], init_azimut = 0, init_inclination = 0,
                 length = 10, step_incl = 1, step_azimuth = 2, angle_constraint_per_m = 1,
                 random_start_point = False, steps_cube_ahead = 10
                 ):
        self.action_space = Discrete(9)
        self.length = length
        self.prod_map = prod_map
        self.rew_sum = 0
        self.start_point = start_point
        self.random_start_point = random_start_point
        self.steps_cube_ahead = steps_cube_ahead
        if random_start_point:
            self.state = [np.random.randint(20, self.prod_map.shape[0] - 10), np.random.randint(20, self.prod_map.shape[1]), start_point[2]]
        else:
            self.state =  self.start_point

        # observation is our current position plus some cube part
        cube_part = (self.prod_map[self.start_point[0]:self.start_point[0] + self.steps_cube_ahead,
                    self.start_point[1]: self.start_point[1] + self.steps_cube_ahead,
                    self.start_point[2]: self.start_point[2] + self.steps_cube_ahead]).flatten()

        self.observation = np.hstack((np.array(self.state), cube_part))
        self.traj = [self.state]
        self.step_incl = step_incl
        self.step_azimuth = step_azimuth
        self.incl_l = [init_inclination]
        self.azi_l = [init_azimut]
        self.angle_constraint = angle_constraint_per_m
        self.plotting_R = []
        print('Environment initialized')
        self.done = False

        self.items = [[-self.step_incl, self.step_azimuth], [-self.step_incl, -self.step_azimuth], [-self.step_incl, 0], [0 ,0],
                      [0, -self.step_azimuth], [0, self.step_azimuth],
                      [self.step_incl, self.step_azimuth], [self.step_incl, 0], [self.step_incl, -self.step_azimuth]]

        self.x = np.linspace(0, self.prod_map.shape[0] - 1, self.prod_map.shape[0])
        self.y = np.linspace(0, self.prod_map.shape[1] - 1, self.prod_map.shape[1])
        self.z = np.linspace(0, self.prod_map.shape[2] - 1, self.prod_map.shape[2])
        self.points = (self.x, self.y, self.z)

    def _action_reward(self, action):
        k = 0
        if len(self.incl_l) < 3:
            self.incl_l.append(self.incl_l[-1] + self.items[action][0])
            self.azi_l.append(self.azi_l[-1] + self.items[action][1])
        else:
            incl2 = self.incl_l[-1]
            azi2 = self.azi_l[-1]
            incl1 = self.incl_l[-2]
            azi1 = self.azi_l[-2]
            k = 0
            dls_val = np.linalg.norm(incl1 - incl2) + np.linalg.norm(azi1 - azi2)
            if dls_val >= self.angle_constraint * self.length:
                k = 1
        if (self.state[0] >= self.prod_map.shape[0] - 30) or (self.state[0] <= 20) or \
                (self.state[1] >= self.prod_map.shape[1] - 30) or (self.state[1] <= 20) or \
                (self.state[2] >= self.prod_map.shape[2] - 30) or (self.state[2] <= 20):
            k = 1
            self.done = True
        if k == 1:
            next_incl_diff = 0
            next_azi_diff = 0
            # high penalty for going out of boundary
            penalty = -100
        else:
            next_incl_diff = self.items[action][0]
            next_azi_diff = self.items[action][1]
            penalty = 0

        self.incl_l.append(self.incl_l[-1] + next_incl_diff)
        self.azi_l.append(self.azi_l[-1] + next_azi_diff)
        vec_diff = self.get_vec(self.incl_l[-1], self.azi_l[-1])

        self.state = [self.state[0] + vec_diff[0], self.state[1] + vec_diff[1], self.state[2] + vec_diff[2]]
        # productivity potential at this point normalized be step size dogleg severity
        if len(self.incl_l) < 4:
            OFV = interpn(self.points, self.prod_map, self.state, method = 'nearest')/np.linalg.norm(vec_diff)
        else:
            OFV = interpn(self.points, self.prod_map, self.state, method='nearest') / np.linalg.norm(vec_diff) - dls_val * 10
        OFV += penalty
        self.traj.append(self.state)
        return OFV

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


    def step(self, action):
        """Run one timestep of the environment's dynamics.
            When end of episode is reached, you are responsible for calling :meth:`reset` to reset this environment's state.
            Accepts an action and returns either a tuple `(observation, reward, terminated, truncated, info)`.
            Args:
                action (ActType): an action provided by the agent
            Return:
              []
              tuple[np.array, float, bool, bool, dict]
        """
        self.done = False
        info = {}
        rw = self._action_reward(action)
        # if (self.state[2] <= 20) or (self.state[1] <= 20) or \
        #     (self.state[0] <= 20):
        #     self.done = True

        # if (self.state[2] >= self.prod_map.shape[2] - 20) or (self.state[1] <= self.prod_map.shape[1] - 20) or \
        #     (self.state[0] <= self.prod_map.shape[0] - 20):
        #     self.done = True
      #  print(self.state)
        cube_part = (self.prod_map[int(self.state[0]):int(self.state[0]) + self.steps_cube_ahead,
                     int(self.state[1]): int(self.state[1]) + self.steps_cube_ahead,
                     int(self.state[2]): int(self.state[2]) + self.steps_cube_ahead]).flatten()
       # print(cube_part.shape)
        self.observation = np.hstack((np.array(self.state), cube_part))
        return self.observation, rw, self.done, info

    def render(self, action, rw, step):
        """Compute the render frames as specified by render_mode attribute during initialization of the environment.
        """
        pass

    def reset(self, init_azimut=0, init_inclination=0):
        """Resets the environment to an initial state and returns the initial observation.
            This method can reset the environment's random number generator(s) if ``seed`` is an integer or
            if the environment has not yet initialized a random number generator.
            If the environment already has a random number generator and :meth:`reset` is called with ``seed=None``,
            the RNG should not be reset. Moreover, :meth:`reset` should (in the typical use case) be called with an
            integer seed right after initialization and then never again.
        """
        self.rew_sum = 0
        if self.random_start_point:
            self.state = [np.random.randint(20, self.prod_map.shape[0] - 10), np.random.randint(20, self.prod_map.shape[1]),
                          np.random.randint(20, self.prod_map.shape[2] - 10)]
        else:
            self.state = self.start_point
        self.traj = [self.state]
        self.incl_l = [init_inclination]
        self.azi_l = [init_azimut]
        self.traj = [self.state[1]]

        cube_part = (self.prod_map[self.state[0]:self.state[0] + self.steps_cube_ahead,
                     self.state[1]: self.state[1] + self.steps_cube_ahead,
                     self.state[2]: self.state[2] + self.steps_cube_ahead]).flatten()

        self.observation = np.hstack((np.array(self.state), cube_part))

        return self.observation