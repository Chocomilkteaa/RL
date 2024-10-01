import os
import cv2

from collections import deque
import random
import numpy as np
import gymnasium as gym

import torch

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
    def sample(self, batch_size):
            return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)
    def reset(self):
        self.memory.clear()

class Model(torch.nn.Module):
    def __init__(self, number_of_state, number_of_action, hidden_size):
        super(Model, self).__init__()
        self.layer1 = torch.nn.Linear(number_of_state, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, number_of_action)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        return self.layer3(x)
    
class DQN(object):
    def __init__(self, env, size, number_of_state, number_of_action, 
                hidden_size=128, batch_size=128, memory_size=10000,
                max_epsisode=1000, explore_rate=0.99, learning_rate=1e-4, update_rate=0.005,  discount_rate=0.99,
                test_iter=10, result_path='./Result', result_name='video'):
        self.env = env
        self.size = size
        self.number_of_state = number_of_state
        self.number_of_action = number_of_action

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.memory = memory_size

        self.max_epsisode = max_epsisode
        self.explore_rate = explore_rate
        self.learning_rate = learning_rate
        self.update_rate = update_rate
        self.discount_rate = discount_rate

        self.test_iter = test_iter

        self.result_path = result_path
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        
        self.result_name = result_name

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.memory = ReplayMemory(10000)

        self.policy_net = DQN(number_of_state, number_of_action).to(self.device)
        self.target_net = DQN(number_of_state, number_of_action).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.MSELoss()

    def sampleTransition(self):
        transitions = self.memory.sample(self.batch_size)

        state_batch = []
        action_batch = []
        reward_batch = []
        n_done_batch = []
        n_done_next_state_batch = []

        for state, action, reward, next_state in transitions:
            state_batch.append(torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0))
            reward_batch.append(torch.tensor([reward], dtype=torch.float32, device=self.device))
            n_done_batch.append((next_state is not None))
            if next_state is not None:
                n_done_next_state_batch.append(torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0))

        state_batch = torch.cat(state_batch)
        reward_batch = torch.cat(reward_batch)
        n_done_batch = torch.tensor(n_done_batch, dtype=torch.bool)
        n_done_next_state_batch = torch.cat(n_done_next_state_batch)

        return state_batch, reward_batch, n_done_batch, n_done_next_state_batch
    
    def updateTarget(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.update_rate + target_net_state_dict[key]*(1-self.update_rate)
        self.target_net.load_state_dict(target_net_state_dict)

    def optimizePolicy(self):
        if len(self.memory) < self.batch_size:
            return

        state_batch, reward_batch, n_done_batch, n_done_next_state_batch = self.sampleTransition()

        state_action_values = self.policy_net(state_batch).max(1).values

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[n_done_batch] = self.target_net(n_done_next_state_batch).max(1).values

        expected_state_action_values = (next_state_values * self.discount_rate) + reward_batch

        loss = self.criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.updateTarget()

    def dqn(self):
        for episode in range(self.max_epsisode):
            state, info = env.reset(seed = episode)

            explore_rate *= 0.999

            rewards = 0

            while True:
                random_num = np.random.uniform(0,1)

                if random_num > explore_rate:
                    action = self.policy_net(torch.tensor(state, dtype=torch.float32, device='cuda').unsqueeze(0)).max(1).indices.item()
                else:
                    action = env.action_space.sample()

                next_state, reward, terminated, truncated, done = env.step(action)

                rewards += reward

                if terminated:
                    next_state = None

                self.memory.push(state, action, reward, next_state)

                self.optimizePolicy()

                if terminated or truncated:
                    print(f'Episode: {episode} ends with reward {rewards}')
                    break

                state = next_state


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    env = gym.make("CartPole-v1", render_mode='rgb_array')
    observation, info = env.reset()
    frame = env.render()
    size = (frame.shape[1], frame.shape[0])
    number_of_state = env.observation_space.shape[0]
    number_of_action = env.action_space.n

    ql = DQN(env, size, number_of_state, number_of_action,  result_path='./Result/', result_name='vi')
    ql.dqn()