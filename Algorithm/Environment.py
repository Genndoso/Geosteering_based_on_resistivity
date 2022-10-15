import gym
from gym.spaces import Discrete, Box
from gym import Env
import numpy as np

from Algorithm import Greedy_algorithm


class ProdMapEnv(Env):
    def __init__(self, action_space: np.array, obs: np.array, productivity, start_point=[0, 20], step_x=1, step_y=1):
        self.action_space = Discrete(3)

        self.observation_space = obs
        self.prod_map = productivity
        self.rew_sum = 0
        self.start_point = start_point
        self.state = [start_point[0], np.random.randint(0, self.observation_space.shape[1])]
        self.traj = [self.state[1]]
        self.step_x = step_x
        self.step_y = step_y
        print('Environment initialized')
        self.done = False

    def _take_action(self, action):
        if action == 0:
            self.state = [self.state[0] + self.step_x, self.state[1]]

        if action == 1:
            self.state = [self.state[0] + self.step_x, self.state[1] + self.step_y]

        if action == 2:
            self.state = [self.state[0] + self.step_x, self.state[1] - self.step_y]

    def _get_rewarded(self):
        rw = 0
        if self.state[1] == self.observation_space.shape[1] and self.state[0] != self.observation_space.shape[0]:
            self.done = True
            rw -= 10

        if self.state[0] == self.observation_space.shape[0] - 1:
            self.done = True
            rw += 100

        try:
            rw += self.prod_map[self.state[0], self.state[1]] * 10
        except:
            rw -= 10

        self.rew_sum += rw
        return rw

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
        if (self.state[0] == self.observation_space.shape[0]) or \
                + (self.state[1] >= self.observation_space.shape[1]) or (self.state[1] <= 0) or (self.state[0] <= 0):
            obs = self.observation_space[self.state[0], self.state[1]]
            return np.array(obs), rw, self.done, info

        # take action
        self._take_action(action)
        # get reward
        rw = self._get_rewarded()

        # update trajectory
        self.traj.append(self.state[1])

        try:
            obs = self.observation_space[self.state[0], self.state[1]]
        except:
            obs = self.observation_space[self.state[0], self.state[1] - 1]

        return np.array(obs), rw, self.done, info

    def render(self, action, rw, step):

        """Compute the render frames as specified by render_mode attribute during initialization of the environment.
        """
        print(f"Step : {step}\nDistance Travelled : {action}\nReward Received: {rw}")
        print(f"Total Reward : {self.rew_sum}")
        print("=============================================================================")

        gr = Greedy_algorithm.greedy_algorithm_main.greedy_algorithm_main(self.prod_map,
                                                                                             angle_constraint=2,
                                                                                             step_back=3)
        gr.visualization(self.observation_space.T, self.traj)

    def reset(self):
        """Resets the environment to an initial state and returns the initial observation.
            This method can reset the environment's random number generator(s) if ``seed`` is an integer or
            if the environment has not yet initialized a random number generator.
            If the environment already has a random number generator and :meth:`reset` is called with ``seed=None``,
            the RNG should not be reset. Moreover, :meth:`reset` should (in the typical use case) be called with an
            integer seed right after initialization and then never again.
        """
        self.rew_sum = 0
        self.state = [self.start_point[0], np.random.randint(0, self.observation_space.shape[1])]
        self.traj = [self.state[1]]
        obs = self.observation_space[self.state[0], self.state[1]]
        return obs