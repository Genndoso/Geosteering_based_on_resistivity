import torch
from torch import nn
use_cuda = torch.cuda.is_available()
device  = torch.device("cuda" if use_cuda else "cpu")
from ReplayBuffer import ReplayBuffer
import numpy as np
from torch.nn.functional import normalize

class Actor(nn.Module):

    def __init__(self, num_inputs, action_space, hidden_size):
        super(Actor, self).__init__()
        num_outputs = action_space
        self.network = nn.Sequential(nn.Linear(num_inputs, hidden_size),
                                     nn.LayerNorm(hidden_size),
                                     nn.Dropout(),
                                     nn.LeakyReLU(),
                                     nn.Linear(hidden_size, hidden_size),
                                     nn.LayerNorm(hidden_size),
                                     nn.LeakyReLU(),
                                     nn.Linear(hidden_size, hidden_size),
                                     nn.LayerNorm(hidden_size),
                                     nn.LeakyReLU(),
                                     )
        self.tanh = nn.Tanh()
        self.mu = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
      #  x = normalize(x, dim = 0)
        x = self.network(x)
        x = self.mu(x)
        x = self.tanh(x)
        return x

    def get_action(self, state):
        state = torch.FloatTensor(state).to(device)  # .unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()  # [0, 0]


class Critic(nn.Module):

    def __init__(self, num_inputs, action_space, hidden_size):
        super(Critic, self).__init__()
        # Defining the first Critic neural network
        self.layer_1 = nn.Linear(num_inputs + action_space, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        self.layer_3 = nn.Linear(hidden_size, 1)
        # Defining the second Critic neural network
        self.layer_4 = nn.Linear(num_inputs + action_space, hidden_size)
        self.layer_5 = nn.Linear(hidden_size, hidden_size)
        self.layer_6 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x, u):
       # x = normalize(x, dim=0)
       # u = normalize(u, dim=0)
        xu = torch.cat([x, u], 1)
        # Forward-Propagation on the first Critic Neural Network
        x1 = self.relu(self.layer_1(xu))
        x1 = self.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        # Forward-Propagation on the second Critic Neural Network
        x2 = self.relu(self.layer_4(xu))
        x2 = self.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = self.relu(self.layer_1(xu))
        x1 = self.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1



class DDPG:
    def __init__(self, state_dim, action_dim, hidden_size, action_space, critic_lr, actor_lr, replay_buffer_size = 100000, loss_type = 'MSE'):
        # actor
        self.actor = Actor(state_dim, action_dim, hidden_size).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_size).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = actor_lr)
        self.action_space = action_space
        # critic
        self.critic = Critic(state_dim, action_dim, hidden_size).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_size).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = critic_lr)

        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        # Loss type
        if loss_type == 'MSE':
            self.loss = nn.MSELoss()
        elif loss_type == 'Huber':
            self.loss = nn.HuberLoss()

    def hard_update(target_params, source_params,  tau):
        for target_param, param in zip(target_params, source_params):
            target_param.data.copy_(param.data)

    def soft_update(self, target_params, source_params, tau):
        for target_param, param in zip(target_params, source_params):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def update_models(self, batch_size = 128, gamma = 0.99, tau  = 0.02, policy_noise = 2, noise_clip = 0.5):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)


        #  From the next state s’, the Actor target plays the next action a’
        next_action = self.actor_target(next_state)

        # define noise
        noise = torch.Tensor(action).data.normal_(0, policy_noise).to(device)
        noise = noise.clamp(-noise_clip, noise_clip)

        next_action[:, 0] = (next_action[:, 0] + noise[:, 0]).clamp(self.action_space.low[0], self.action_space.high[0])
        next_action[:, 1] = (next_action[:, 1] + noise[:, 1]).clamp(self.action_space.low[1], self.action_space.high[1])
        # The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        # We keep the minimum of these two Q-values: min(Qt1, Qt2)
        target_Q = torch.min(target_Q1, target_Q2)

        # We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
        target_Q = reward + ((1 - done) * gamma * target_Q).detach()


        # The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
        current_Q1, current_Q2 = self.critic(state, action)

        #  We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
        critic_loss = self.loss(current_Q1, target_Q) + self.loss(current_Q2, target_Q)

        #  We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # Still once every two iterations, we update the weights of the Actor target by polyak averaging
        self.soft_update(self.actor_target.parameters(), self.actor.parameters(), tau = tau)

        # Still once every two iterations, we update the weights of the Critic target by polyak averaging
        self.soft_update(self.critic_target.parameters(), self.critic.parameters(), tau = tau)


    # Making a save method to save a trained model
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    # Making a load method to load a pre-trained model
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))