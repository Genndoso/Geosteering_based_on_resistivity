from geostering_env import Geosteering_env
from replay_buffer import ReplayBuffer
import torch
from torch import nn
import numpy as np
import os

class Agent:
    def __init__(self, type, gamma, lr, n_actions, n_states,
                 batch_size, mem_size, replace, saved_dir, epsilon, env_name):

        self.type = type
        self.gamma = gamma
        self.action_space = np.arange(n_actions)
        self.batch_size = batch_size
        self.replace_num = replace
        self.epsilon = epsilon
        self.learn_idx = 0
        self.loss_plot = 0
        self.running_loss = 0
        self.loss = nn.HuberLoss()
        #   self.loss = nn.MSELoss()
        self.memory = ReplayBuffer(mem_size, n_states)

        self.Q_eval = DeepQNetwork(lr = lr ,out_dims=n_actions ,input_dims=n_states,
                                   name=env_name +'.pth', saved_dir=saved_dir,
                                   hid_dim = 128)
        self.Q_next = DeepQNetwork(lr=lr ,out_dims=n_actions ,input_dims=n_states,
                                   name=env_name +'_q_next.pth', saved_dir=saved_dir,
                                   hid_dim = 128)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor(np.array(observation)).to(self.Q_eval.device)
            action = torch.argmax(self.Q_eval.forward(state)).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def save_models(self):
        self.Q_eval.save_checkpoint()

    def learn(self):
        if self.memory.mem_idx < self.batch_size:
            return

        if self.learn_idx % self.replace_num == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)

        batch_index = np.arange(self.batch_size, dtype=np.int32)



        if self.type == 'dqn':
            q_eval = self.Q_eval.forward(states)[batch_index, actions]
            q_next = self.Q_next.forward(next_states).max(dim=1)[0]

            q_next[dones] = 0.0

            q_target = rewards + self.gamma * q_next
            loss = self.loss(q_target, q_eval).to(self.Q_eval.device)

        elif self.type == 'doubledqn':

            q_pred = self.Q_eval.forward(states)[batch_index, actions]
            q_next = self.Q_next.forward(next_states)
            q_eval = self.Q_eval.forward(next_states)

            max_actions = torch.argmax(q_eval, dim=1)
            q_next[dones] = 0.0

            q_target = rewards + self.gamma * q_next[batch_index, max_actions]


            loss = self.loss(q_target.clone(), q_pred.clone())  # .to(self.Q_eval.device)


        self.Q_eval.optimizer.zero_grad()

        loss.backward()
        self.Q_eval.optimizer.step()


        self.running_loss += loss.item()
        self.learn_idx += 1


class DeepQNetwork(nn.Module):
    def __init__(self, lr, out_dims, input_dims, name, saved_dir, hid_dim):
        super().__init__()
        self.checkpoint_file = os.path.join(saved_dir, name)

        self.layers = nn.Sequential(
            nn.Linear(input_dims, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dims),
        )

        self.optimizer =  torch.optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cpu')
        self.to('cpu')

    def forward(self, state):
        state = state.type(torch.FloatTensor)
        x = self.layers(state)
        q_val = ((x))
        return q_val

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
