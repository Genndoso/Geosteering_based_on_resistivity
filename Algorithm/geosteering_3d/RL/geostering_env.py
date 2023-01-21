from gym.spaces import Discrete, Box
from gym import Env
import os
import numpy as np
from scipy.interpolate import interpn
import matplotlib.pyplot as plt


class ObservationGeosteering():
    def __init__(self):
        self.shape = (6,)


class ActionSpaceGeosteering():
    def __init__(self):
        self.shape = (2,)
        self.action_space = Box(low = np.array([0, -20]), high = np.array([90, 180]), shape=(2,))


class Geosteering_env(Env):
    def __init__(self,  prod_map: np.array, start_point=[0, 20, 20], init_azimut = 0, init_inclination = 0,
                 length = 10, angle_constraint_per_m = 1,
                 random_start_point = False, random_angles = False, steps_cube_ahead = 10
                 ):
        self.action_space = Box(low = np.array([0, -20]), high = np.array([90, 180]), shape=(2,))
        self.observation_space = ObservationGeosteering()
        self.length = length
        self.prod_map = prod_map
        self.rew_sum = 0
        self.start_point = start_point
        self.steps_cube_ahead = steps_cube_ahead
        if random_start_point:
            self.pos_state = [np.random.randint(20, self.prod_map.shape[0] - 30), np.random.randint(30, self.prod_map.shape[1]) - 30, np.random.randint(30, self.prod_map.shape[2]) - 30]
        else:
            self.pos_state = self.start_point

        # observation is our current position plus some cube part
        cube_part = (self.prod_map[self.start_point[0]:self.start_point[0] + self.steps_cube_ahead,
                    self.start_point[1]: self.start_point[1] + self.steps_cube_ahead,
                    self.start_point[2]: self.start_point[2] + self.steps_cube_ahead]).flatten()

        self.observation = np.hstack((np.array(self.pos_state), cube_part))

        if random_angles:
            self.incl_l = [np.random.randint(0, 90)]
            self.azi_l = [np.random.randint(0, 180)]
        else:
            self.incl_l = [init_inclination]
            self.azi_l = [init_azimut]
        self.angle_constraint = angle_constraint_per_m
        self.plotting_R = []
        self.random_start_point = random_start_point
        print('Environment initialized')
        self.done = False

        self.x = np.linspace(0, self.prod_map.shape[0] - 1, self.prod_map.shape[0])
        self.y = np.linspace(0, self.prod_map.shape[1] - 1, self.prod_map.shape[1])
        self.z = np.linspace(0, self.prod_map.shape[2] - 1, self.prod_map.shape[2])
        self.points = (self.x, self.y, self.z)

        self.state = np.array([self.pos_state[0], self.pos_state[1], self.pos_state[2], self.incl_l[-1], self.azi_l[-1], self.angle_constraint])

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


    def _step_reward(self, incl, azi):

        penalty = 0
        closeness_rew = 0
        vec_diff = self.get_vec(incl, azi)

        state_new = np.array([self.state[0] + vec_diff[0], self.state[1] + vec_diff[1], self.state[2] + vec_diff[2], self.incl_l[-1], self.azi_l[-1], self.angle_constraint])
        state_interp = [state_new[0], state_new[1], state_new[2]]
        # dogleg severity constraint
        if len(self.incl_l) > 3:
            dls_val = np.linalg.norm(incl - self.state[3]) + np.linalg.norm(azi - self.state[4])
            if dls_val >= self.angle_constraint * self.length:
                penalty += dls_val * 0.01

        # constraint for high length action
        length_constraint = np.linalg.norm(vec_diff)
        # get productivity potential
        OFV = interpn(self.points, self.prod_map, state_interp, method='nearest') * 10 / length_constraint

        # get the longest axis
        axis_idx = np.array([self.prod_map.shape[0], self.prod_map.shape[1], self.prod_map.shape[2]]).argmax()
        # reward for approaching right border of the cube
      #  length_rew = 1 - (self.prod_map.shape[axis_idx] - state_new[axis_idx] / self.prod_map.shape[axis_idx] )
        if state_new[axis_idx] >= self.prod_map.shape[axis_idx] - 30:
            closeness_rew = 100
       # print(OFV, penalty, length_constraint)
        step_rew = (OFV  - penalty) + closeness_rew #length_rew)*0.01
        return step_rew, state_new

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
        rw = 0

        if (self.state[0] >= self.prod_map.shape[0] - 20) or (self.state[0] <= 20) or \
                (self.state[1] >= self.prod_map.shape[1] - 20) or (self.state[1] <= 20) or \
                (self.state[2] >= self.prod_map.shape[2] - 20) or (self.state[2] <= 20):
            self.done = True
            rw -= np.array([10])
        if not self.done:
            rw, self.state = self._step_reward(action[0], action[1])

        self.incl_l.append(action[0])
        self.azi_l.append(action[1])

        #append plotting reward
        self.plotting_R.append(rw)

        # cube_part = (self.prod_map[int(self.state[0]):int(self.state[0]) + self.steps_cube_ahead,
        #              int(self.state[1]): int(self.state[1]) + self.steps_cube_ahead,
        #              int(self.state[2]): int(self.state[2]) + self.steps_cube_ahead]).flatten()
        #
        # self.observation = np.hstack((np.array(self.state), cube_part))
        return self.state, rw, self.done, info

    def render(self, action, rw, step):
        """Compute the render frames as specified by render_mode attribute during initialization of the environment.
        """
        plt.style.use('seaborn')
        fig, ax = plt.subplots(1,1, figsize = (20,10))
        plt.plot(self.plotting_R, color = 'r')

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
            self.pos_state = [np.random.randint(20, self.prod_map.shape[0] - 10), np.random.randint(20, self.prod_map.shape[1]),
                          np.random.randint(20, self.prod_map.shape[2] - 10)]
        else:
            self.pos_state = self.start_point
        self.traj = [self.state]
        self.incl_l = [init_inclination]
        self.azi_l = [init_azimut]
        self.traj = [self.state[1]]

        # cube_part = (self.prod_map[self.state[0]:self.state[0] + self.steps_cube_ahead,
        #              self.state[1]: self.state[1] + self.steps_cube_ahead,
        #              self.state[2]: self.state[2] + self.steps_cube_ahead]).flatten()
        #
        # self.observation = np.hstack((np.array(self.state), cube_part))

        self.state = np.array([self.pos_state[0], self.pos_state[1], self.pos_state[2], self.incl_l[-1], self.azi_l[-1], self.angle_constraint])
        done = False
        return self.state, done