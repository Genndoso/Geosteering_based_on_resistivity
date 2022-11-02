import numpy as np
#import Greedy_algorithm
import matplotlib.pyplot as plt
from .Greedy_algorithm import greedy_algorithm_main

class q_learning:

    def __init__(self, map, q_goal, alpha, gamma, epsilon, n_episodes,
                 n_iterations):  # Initialize Q-learning parameters
        # Actions: 0 right, 1 up, 2 down
        self.actions = {0: np.array([1, 0]), 1: np.array([1, 1]), 2: np.array([1, -1])}
        self.map = map
        self.Q = np.zeros((map.shape[0], map.shape[1], 3))
        self.policy = np.zeros((map.shape[0], map.shape[1]))
        self.v = np.zeros((map.shape[0], map.shape[1]))
        self.q_goal = np.array(q_goal)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_iterations = n_iterations
        self.n_episodes = n_episodes

        self.R = 0
        self.plotting_R = []

    def dynamics(self, a, s):  # Dynamics Function
        r = 0
        s_prime = s + self.actions[a]

        if s_prime[1] + 1 == self.map.shape[1]:
            s_prime = s

        r += self.map[s_prime[0], s_prime[1]]

        #  if s_prime[0] + 1 == self.map.shape[0] or s_prime[1] + 1 == self.map.shape[1]:

        if s_prime[0] + 1 == self.q_goal[0][0]:
            r += 10
        else:
            r += -0.2
        return s_prime, r

    def epsilon_greedy_policy(self, s):  # Epsilon greedy policy (choose a random action with probability epsilon)
        if np.random.sample() < self.epsilon:
            return int(np.random.randint(0, 3))
        else:
            return int(np.argmax(self.Q[s[0], s[1], :]))

    def initialize_s(self):  # Initialize to a random state that is free and its not the goal state

        #  x_s = np.random.choice(self.map.shape[0] - 1, 1)
        x_s = 0
        y_s = np.random.choice(self.map.shape[1] - 1, 1)

        while self.map[x_s, y_s] == 1 or (
                x_s == self.q_goal[0][0] and y_s not in [self.q_goal[i][1] for i in range(0, len(self.q_goal))]):
            x_s = np.random.choice(self.map.shape[0], 1)
            y_s = np.random.choice(self.map.shape[1], 1)

        return np.append(x_s, y_s)

    def episode(self, angle_constraint = 1.5):  # Episode execution for n_iterations
        self.R = 0
        s = self.initialize_s()
        traj = []
        angle_constraint = angle_constraint
        gr = greedy_algorithm_main(self.map, angle_constraint = angle_constraint, step_back=3)

        for i in range(0, self.n_iterations):

            a = self.epsilon_greedy_policy(s)

            s_prime, r = self.dynamics(a, s)
            traj.append(s_prime[1])

            # if i > 10:
            #     deg = gr.deg_calc(traj)

            #     if deg >= angle_constraint:
            #         r += -1

            self.R += r

            self.Q[s[0], s[1], a] = self.Q[s[0], s[1], a] + self.alpha * (
                    r + self.gamma * np.max(self.Q[s_prime[0], s_prime[1], :]) - self.Q[s[0], s[1], a])

            s = s_prime

            # if np.array_equal(s_prime, self.q_goal):
            if s_prime[0] + 1 == self.q_goal[0][0]:
                break

        return self.Q

    def optimal_policy(self):  # Retrieve the optimal policy from Q(s,a)
        self.policy = np.argmax(self.Q, axis=2)

    def value_function(self):  # Retrieve the optimal value function from from Q(s,a)
        self.v = np.max(self.Q, axis=2)
       # np.savez_compressed('value_f.npz', self.v)
        #  self.v = np.where(self.obstacles == False, self.v, self.obs_thres)
        return self.v

    def execution(self):
        # Execute n_episodes and every 200 episodes stop training in order
        # to retrieve the average reward for 100 episodes, then resume training

        for i in range(1, self.n_episodes + 1):

            self.episode()

            if i % 200 == 0:
                cum_R = 0
                self.alpha = 0
                self.epsilon = 0

                for j in range(1, 101):
                    self.episode()
                    cum_R += self.R

                self.plotting_R.append(cum_R / 100)

                self.alpha = 0.1
                self.epsilon = 0.3

            print(f'Episod {i} is passed')

        self.value_function()
        self.optimal_policy()

    def plotting_effectiveness(
            self):  # Plotting Effectiveness function, x axis: number of episodes, y axis: avg. reward
        plt.figure(figsize=(10, 8))
        plt.plot(range(0, self.n_episodes, 200), self.plotting_R, linewidth=3)
        plt.ylabel('Avg. Reward')
        plt.xlabel('Number of Episodes')
        plt.title('Effectiveness plot')

    def plotting_value_function(self):  # Show every gridmap state
        plt.matshow(self.v, cmap="jet")
        plt.colorbar()
        plt.scatter(self.q_goal[1], self.q_goal[0])
        plt.title('Value Function Plot')

    def plotting_policy(self):  # Plotting the optimal policy
        ''' Plotting the optimal policy the agent has to follow in order to achieve the goal, in this case plotting
        an arrow in the direction of the optimal action to take at every state of the environment (grid map)'''

        plt.matshow(self.v, cmap="jet")
        plt.colorbar()
        for i in range(0, self.policy.shape[0]):
            for j in range(0, self.policy.shape[1]):
                if i == self.q_goal[0][0] and j in [i for i in range(0, self.q_goal.shape[0])]:
                    plt.scatter(self.q_goal[1], self.q_goal[0])
                elif self.policy[i, j] == 0 and self.v[i, j] != self.obs_thres:
                    plt.scatter(j, i, marker=">")
                elif self.policy[i, j] == 1 and self.v[i, j] != self.obs_thres:
                    plt.scatter(j, i, marker="^")
                elif self.policy[i, j] == 2 and self.v[i, j] != self.obs_thres:
                    plt.scatter(j, i, marker="v")

        plt.title('Optimal policy pi*')

    def q_learning_plot(self, start_point=[0, 150], angle_constraint = 1.5):
        # plotting trajectory
        # no angle constraint taken into account
        traj = [start_point[1]]
        OFV = 0
        gr = greedy_algorithm_main(self.map, angle_constraint = angle_constraint, step_back = 3)
        angle_constraint = 200
        for i in range(0, self.policy.shape[0] - 1):

            if i > 10:
                deg = gr.deg_calc(traj)

                if deg >= angle_constraint:
                    traj.append(traj[i])
                    OFV += self.map[i, traj[i]]
                    continue

            pol = self.policy[i, traj[i]]
            if pol == 0:
                traj.append(traj[i])
            elif pol == 1:
                traj.append(traj[i] + 1)
            elif pol == 2:
                traj.append(traj[i] - 1)
            OFV += self.map[i, traj[i]]

        print(round(OFV, 2))
        return traj