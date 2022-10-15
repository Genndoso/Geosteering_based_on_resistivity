import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque


import itertools
import time
import os
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

from torch.utils.tensorboard import SummaryWriter
from Environment import ProdMapEnv

# Initialize hyperparameters
GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 10000
MIN_REPLAY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 1000

writer = SummaryWriter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# actor, critic neural networks
dataset = np.copy(dataset_sc)
critic_layer_size = 256
critic = nn.Sequential(nn.Linear(dataset[0][0].shape[0], critic_layer_size),
                       nn.LeakyReLU(),
                       nn.Linear(critic_layer_size, critic_layer_size),
                       nn.LeakyReLU(),
                       nn.Linear(critic_layer_size, critic_layer_size),
                       nn.LeakyReLU(),
                       nn.Linear(critic_layer_size, critic_layer_size),
                       nn.LeakyReLU(),
                       nn.Linear(critic_layer_size, critic_layer_size),
                       nn.LeakyReLU(),
                       nn.Linear(critic_layer_size, critic_layer_size),
                       nn.LeakyReLU(),
                       nn.Linear(critic_layer_size, critic_layer_size),
                       nn.LeakyReLU(),
                       nn.Linear(critic_layer_size, 1),
                       )
actor_layer_size = 256
actor = nn.Sequential(
    nn.Linear(dataset[0][0].shape[0], actor_layer_size),
    nn.LeakyReLU(),
    nn.Linear(actor_layer_size, actor_layer_size),
    nn.LeakyReLU(),
    nn.Linear(actor_layer_size, actor_layer_size),
    nn.LeakyReLU(),
    nn.Linear(actor_layer_size, actor_layer_size),
    nn.LeakyReLU(),
    nn.Linear(actor_layer_size, actor_layer_size),
    nn.LeakyReLU(),
    nn.Linear(actor_layer_size, actor_layer_size),
    nn.LeakyReLU(),
    nn.Linear(actor_layer_size, 3),
    nn.Softmax()
)


class CriticTD(torch.nn.Module):
    def __init__(self, actor, critic, observation_space, prod_map):
        super(self.__class__, self).__init__()
        self.critic = critic.to(device)
        self.actor = actor.to(device)
        self.discount = 0.99

        self.transition = ProdMapEnv(action_space=Discrete(3), obs=observation_space, productivity=prod_map)
        self.loss = torch.nn.MSELoss()
        self.td_steps = 10

    def forward(self, state):
        with torch.no_grad():
            next_state = torch.Tensor(state)
            accumulated_discount = 1.0
            td_target = torch.zeros((state.shape[0], 1)).to(device)
            for _ in range(self.td_steps):
                action = torch.argmax(actor(next_state))

                next_transition = self.transition.step(action)

                next_state = torch.Tensor(next_transition[0])
                reward = next_transition[1]
                td_target += accumulated_discount * reward
                accumulated_discount *= self.discount
            td_target += accumulated_discount * critic(next_state)
        value = critic(state)
        temporal_difference_loss = self.loss(value, td_target)

        return temporal_difference_loss

    def parameters(self):
        return self.critic.parameters()


class ActorImprovedValue(torch.nn.Module):
    def __init__(self, actor, critic, observation_space, prod_map):
        super(self.__class__, self).__init__()
        self.critic = critic.to(device)
        self.actor = actor.to(device)
        self.transition = ProdMapEnv(action_space=Discrete(3), obs=observation_space, productivity=prod_map)
        self.discount = 0.99

    def forward(self, state):
        action = torch.argmax(actor(state)).to(device)

        next_transition = self.transition.step(action)
        next_state = torch.Tensor(next_transition[0])
        reward = next_transition[1]

        improved_value = reward + self.discount * self.critic(next_state)
        return -improved_value.mean()

    def parameters(self):
        return self.actor.parameters()

    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        q_vals = self(obs_t.unsqueeze(0))

        max_q_index = torch.argmax(q_vals, dim=1)[0]

        action = max_q_index.detach().item()
        return action



# define Actor-Critic and optimizers
critic_temporal_difference = CriticTD(actor, critic)
actor_improved_value = ActorImprovedValue(actor, critic)


optimizer_critic_kind = torch.optim.Adam
optimizer_critic_parameters = {
    "lr" : 2e-4,
    "weight_decay" : 0.1
}
optimizer_critic = optimizer_critic_kind(critic_temporal_difference.parameters(), **optimizer_critic_parameters)



optimizer_actor_kind = torch.optim.Adam
optimizer_actor_parameters = {
    "lr" : 2e-4,
    "weight_decay" : 0.1
}
optimizer_actor = optimizer_actor_kind(actor_improved_value.parameters(), **optimizer_actor_parameters)

# Initialize Replay buffer
replay_buffer = deque(maxlen=BUFFER_SIZE)
raw_buffer = deque([0.0], maxlen=100)
episode_reward = 0.0
env = ProdMapEnv(action_space=Discrete(3), obs=dataset_sc, productivity=P[600:1200, :])
obs = env.reset()
for _ in tqdm(range(MIN_REPLAY_SIZE)):
    action = env.action_space.sample()

    new_obs, raw, done, _ = env.step(action)

    transition = (obs, action, raw, done)
    replay_buffer.append(transition)

    if done:
        obs = env.reset()

obs = env.reset()



def training_TD(epochs = 20, iterations = 1000):
    # Main training loop
    EPOCHS = epochs
    ITERATIONS = iterations
    for epoch in tqdm(range(EPOCHS), 'Main training loop'):
        total_TD_loss = []
        total_actor_loss = []
        for iterations in (range(ITERATIONS)):
            epsilon = np.interp(iterations, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
            rnd_sample = random.random()

            if rnd_sample <= epsilon:
                action = env.action_space.sample()
            else:
                action = actor_improved_value.actor(torch.Tensor(obs).flatten())
                action = int(torch.argmax(action))

            new_obs, raw, done, _ = env.step(action)
            try:
                action = int(torch.argmax(action))
            except:
                pass
            transition = (new_obs, action, raw, done)
            replay_buffer.append(transition)
            obs = new_obs

            episode_reward += raw

            if done:
                obs = env.reset()
                raw_buffer.append(episode_reward)
                episode_reward = 0.0
                print('done')
                continue

            # Start Gradient Step
            transitions = random.sample(replay_buffer, BATCH_SIZE)
            for i in range(0, len(transitions)):
                if type(transitions[i][0]) == torch.Tensor:
                    transitions[i][0] = transitions[i][0].detach().numpy().astype(np.float64)
                if type(transitions[i][1]) == torch.Tensor:
                    transitions[i][1] = transitions[i][1].detach().numpy().astype(np.float64)

            obses = np.asarray([t[0] for t in transitions], dtype=np.float64)
            actions = np.asarray([t[1] for t in transitions], dtype=np.int64)
            rews = np.asarray([t[2] for t in transitions], dtype=np.float64)
            dones = np.asarray([t[3] for t in transitions], dtype=np.float64)
            # new_obses = np.asarray([t[4] for t in transitions],dtype=np.float64)

            obses_t = torch.as_tensor(obses, dtype=torch.float32)
            actions_t = torch.as_tensor(actions, dtype=torch.int64)  # .unsqueeze(-1)
            rews_t = torch.as_tensor(rews, dtype=torch.float32)  # .unsqueeze(-1)
            dones_t = torch.as_tensor(dones, dtype=torch.float32)  # .unsqueeze(-1)
            # new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

            # Compute targets

            # critic temporal difference loss

            TD_loss = critic_temporal_difference(obses_t).to(device)
            optimizer_critic.zero_grad()
            TD_loss.backward()
            optimizer_critic.step()

            # actor loss
            # Compute loss
            actor_loss = actor_improved_value(obses_t).to(device)
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            total_TD_loss.append(TD_loss.detach().numpy())
            total_actor_loss.append(actor_loss.detach().numpy())

            # Gradient desc step

            # Logging
            if iterations % 200 == 0:
                # # tensorboard logging
                writer.add_scalar("TD loss", np.array(total_TD_loss).mean())
                writer.add_scalar("Actor loss", np.array(total_actor_loss).mean())
                writer.add_scalar("Rewards", np.mean(rews))

                print(f'Epochs: {epoch}:', 'Step', iterations)
                print('Avg reward', round(np.mean(rews), 3))

                print('Avg temporal difference loss:', round(np.array(total_TD_loss).mean(), 3))





