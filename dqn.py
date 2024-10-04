import os
import cv2

from collections import namedtuple, deque
import random
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'n_done'))

class ReplayMemory(object):
    def __init__(self, capacity, device):
        self.memory = deque([], maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, n_done):
        transition = Transition(torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0),
                                torch.tensor([[action]], dtype=torch.int64, device=self.device),
                                torch.tensor([reward], dtype=torch.float32, device=self.device),
                                torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0),
                                torch.tensor([n_done], dtype=torch.bool, device=self.device))
        self.memory.append(transition)

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
                batch_size=128, hidden_size=128, memory_size=10000,
                max_epsisode=1000, explore_rate=0.99, learning_rate=1e-4, update_rate=0.005,  discount_rate=0.99,
                test_iter=10, result_path='./Result', result_name='video', frame_rate=30):
        self.env = env
        self.size = size
        self.number_of_state = number_of_state
        self.number_of_action = number_of_action

        self.batch_size = batch_size

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

        self.frame_rate = frame_rate

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.memory = ReplayMemory(memory_size, self.device)

        self.policy_net = Model(number_of_state, number_of_action, hidden_size).to(self.device)
        self.target_net = Model(number_of_state, number_of_action, hidden_size).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.MSELoss()

    def sampleTransition(self):
        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        n_done_batch = torch.cat(batch.n_done)

        return state_batch, action_batch, reward_batch, next_state_batch, n_done_batch
    
    def updateTarget(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.update_rate + target_net_state_dict[key] * (1 - self.update_rate)
        self.target_net.load_state_dict(target_net_state_dict)

    def optimizePolicy(self):
        if len(self.memory) < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, n_done_batch = self.sampleTransition()

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[n_done_batch] = self.target_net(next_state_batch[n_done_batch]).max(1).values

        expected_state_action_values = ((next_state_values * self.discount_rate) + reward_batch).unsqueeze(1)

        loss = self.criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.updateTarget()

    def getAction(self, state):
        with torch.no_grad():
            action = self.policy_net(torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)).max(1).indices.item()
        return action

    def train(self):
        for episode in range(self.max_epsisode):
            state, info = env.reset(seed = episode)

            self.explore_rate *= 0.999

            rewards = 0

            while True:
                random_num = np.random.uniform(0,1)

                if random_num > self.explore_rate:
                    action = self.getAction(state)
                else:
                    action = env.action_space.sample()

                next_state, reward, terminated, truncated, info = env.step(action)

                rewards += reward

                self.memory.push(state, action, reward, next_state, not (terminated or truncated))

                self.optimizePolicy()

                if terminated or truncated:
                    print(f'Episode: {episode} ends with reward {rewards}')
                    break

                state = next_state

    def test(self):
        trajectory_rewards = []
        for i in range(self.test_iter):
            state, info = self.env.reset(seed=i)
            trajectory_reward = 0
            trajectory_length = 0

            while True:
                action = self.getAction(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)

                trajectory_reward += reward
                trajectory_length += 1

                state = next_state

                if terminated or truncated:
                    print(f'Iteration {i}: length:{trajectory_length}, reward: {trajectory_reward}')

                    trajectory_rewards.append(trajectory_reward)

                    break

        print(f'Reward Mean: {np.mean(trajectory_rewards)}, Std: {np.std(trajectory_rewards)}')
        
        state, info = self.env.reset(seed=np.argmax(trajectory_rewards).item())

        writer = cv2.VideoWriter(os.path.join(self.result_path, f'{self.result_name}.avi'),cv2.VideoWriter_fourcc(*'DIVX'), self.frame_rate, self.size)

        while True:
            action = self.getAction(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            state = next_state

            writer.write(self.env.render())

            if terminated or truncated:
                writer.release()

                break

    def dqn(self):
        self.train()

        self.test()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def getEnvInfo(env):
    observation, info = env.reset()
    frame = env.render()
    size = (frame.shape[1], frame.shape[0])
    if isinstance(env.observation_space, Box):
        number_of_state = env.observation_space.shape[0]
    elif isinstance(env.observation_space, Discrete):
        number_of_state = env.observation_space.n
    else:
        print('this script only works for Box / Discrete observation spaces.')
        exit()
    if isinstance(env.action_space, Discrete):
        number_of_action = env.action_space.n
    else:
        print('this script only works for Discrete action spaces.')
        exit()

    return size, number_of_state, number_of_action


if __name__ == '__main__':
    set_seed(0)

    env = gym.make("CartPole-v1", render_mode='rgb_array')
    
    size, number_of_state, number_of_action = getEnvInfo(env)

    alg = DQN(env, size, number_of_state, number_of_action, batch_size=64,  result_path='./Result/', result_name='dqn')
    alg.dqn()