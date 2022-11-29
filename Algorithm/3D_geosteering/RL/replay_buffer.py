import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, mem_size, input_shape):
        self.mem_size = mem_size
        self.mem_idx = 0

        self.state_memory = np.zeros((self.mem_size, input_shape),
                                     dtype=np.float32)
        self.next_state_memory = np.copy(self.state_memory)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, next_state, done):
        idx = self.mem_idx % self.mem_size
        self.state_memory[idx] = state
        self.next_state_memory[idx] = next_state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = done
        self.mem_idx += 1

    def sample_buffer(self, batch_size):
        mem_size = min(self.mem_idx, self.mem_size)
        batch = np.random.choice(mem_size, batch_size, replace=False)

        states = torch.tensor(self.state_memory[batch]).to(torch.device('cpu'))
        actions = torch.tensor(self.action_memory[batch]).to(torch.device('cpu'))
        rewards = torch.tensor(self.reward_memory[batch]).to(torch.device('cpu'))
        next_states = torch.tensor(self.next_state_memory[batch]).to(torch.device('cpu'))
        terminal = torch.tensor(self.terminal_memory[batch]).to(torch.device('cpu'))

        return states, actions, rewards, next_states, terminal