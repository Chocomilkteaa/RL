import os
import cv2

from collections import deque
import math
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

import matplotlib.pyplot as plt

import torch

class ReplayMemory(object):
    def __init__(self, memory_size, number_of_state, discount_rate, device):
        self.states = torch.zeros((memory_size, number_of_state), dtype=torch.float32, device=device)
        self.actions = torch.zeros(memory_size, dtype=torch.int64, device=device)
        self.rewards = torch.zeros(memory_size, dtype=torch.float32, device=device)
        self.values = torch.zeros(memory_size, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(memory_size, dtype=torch.float32, device=device)
        self.returns = torch.zeros(memory_size, dtype=torch.float32, device=device)
        self.advantages = torch.zeros(memory_size, dtype=torch.float32, device=device)

        self.memory_size = memory_size

        self.discount_rate = discount_rate

        self.start_idx, self.cur_idx = 0, 0

    def push(self, state, action, reward, value, log_prob):
        assert self.cur_idx < self.memory_size

        self.states[self.cur_idx] = state
        self.actions[self.cur_idx] = action
        self.rewards[self.cur_idx] = reward
        self.values[self.cur_idx] = value
        self.log_probs[self.cur_idx] = log_prob

        self.cur_idx += 1

    def done(self, last_value=0):
        self.advantages[self.cur_idx-1] = self.rewards[self.cur_idx-1] + \
            self.discount_rate * last_value - self.values[self.cur_idx-1]
        
        R = self.returns[self.cur_idx-1] = last_value
        
        for i in range(self.cur_idx-2, self.start_idx-1, -1):
            self.advantages[i] = self.rewards[i] + \
                self.discount_rate * self.values[i+1] - self.values[i]
            
            R = self.returns[i] = self.rewards[i] + R * self.discount_rate

        self.start_idx = self.cur_idx


    def get(self):
        print(self.cur_idx, self.memory_size)
        assert self.cur_idx == self.memory_size

        self.start_idx, self.cur_idx = 0, 0

        data = dict(states=self.states,
                    actions=self.actions,
                    advantages=self.advantages,
                    rewards=self.rewards,
                    returns=self.returns,
                    values=self.values,
                    log_probs=self.log_probs)
    
        return {key: value for key, value in data.items()}
    
    def reset(self):
        self.start_idx, self.cur_idx = 0, 0

class Actor(torch.nn.Module):
    def __init__(self, number_of_state, number_of_action, hidden_size):
        super(Actor, self).__init__()
        self.layer1 = torch.nn.Linear(number_of_state, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, number_of_action)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x = torch.nn.functional.softmax(self.layer3(x), dim=1)
        return x
    
    def getActionAndProb(self, state):
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)

        action = dist.sample()

        return action.item(), dist.log_prob(action)
    
    def getProb(self, state, action):
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)

        return dist.log_prob(action)
    
class Critic(torch.nn.Module):
    def __init__(self, number_of_state, hidden_size):
        super(Critic, self).__init__()
        self.layer1 = torch.nn.Linear(number_of_state, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
class PPO(object):
    def __init__(self, env, size, number_of_state, number_of_action, 
                batch_size=128, hidden_size=128, memory_size=10000,
                max_epsisode=1000, actor_learning_rate=1e-4, critic_learning_rate=1e-4, discount_rate=0.99,
                target_score=500, test_iter=10, result_path='./Result', result_name='video', frame_rate=30):
        self.env = env
        self.size = size
        self.number_of_state = number_of_state
        self.number_of_action = number_of_action

        self.batch_size = batch_size

        self.max_epsisode = max_epsisode
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.discount_rate = discount_rate

        self.target_score = target_score

        self.test_iter = test_iter

        self.result_path = result_path
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        
        self.result_name = result_name

        self.frame_rate = frame_rate

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.memory_size = memory_size
        self.memory = ReplayMemory(memory_size, number_of_state, discount_rate, self.device)

        self.actor_net = Actor(number_of_state, number_of_action, hidden_size).to(self.device)
        self.critic_net = Critic(number_of_state, hidden_size).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.critic_learning_rate)

        self.criterion = torch.nn.MSELoss()

        self.eps = np.finfo(np.float32).eps.item()

    def optimizePolicy(self):
        data = self.memory.get()

        batch_idx = np.arange(self.batch_size)
        np.random.shuffle(batch_idx)

        for i in range(int(math.floor(self.memory_size / self.batch_size))):
            minibatch_idx = batch_idx[i*self.batch_size:(i+1)*self.batch_size]
            states = data['states'][minibatch_idx]
            actions = data['actions'][minibatch_idx]
            advantages = data['advantages'][minibatch_idx]
            returns = data['returns'][minibatch_idx]
            log_probs_old = data['log_probs'][minibatch_idx]

            log_probs = self.actor_net.getProb(states, actions)
            
            ratios = torch.exp(log_probs-log_probs_old)
            actor_loss = -torch.min(ratios * advantages, torch.clamp(ratios, 1 - 0.2, 1 + 0.2) * advantages).mean()

            values = self.critic_net.forward(states)

            critic_loss = self.criterion(values.squeeze(), returns)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

    def train(self):
        average_trajectory_reward = deque(maxlen=100)

        for episode in range(self.max_epsisode):
            state, info = self.env.reset(seed = episode)
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            rewards = 0

            for j in range(self.memory_size):
                action, log_prob = self.actor_net.getActionAndProb(state)

                next_state, reward, terminated, truncated, info = env.step(action)

                rewards += reward

                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)

                value = self.critic_net.forward(state)

                self.memory.push(state, action, reward, value, log_prob)

                if terminated or truncated or j == self.memory_size-1:
                    average_trajectory_reward.append(rewards)

                    state, info = self.env.reset(seed = episode)
                    state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

                    if terminated:
                        self.memory.done()
                    else:
                        last_value = self.critic_net.forward(next_state)
                        self.memory.done(last_value)
                else:
                    state = next_state

            self.optimizePolicy()

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

    alg = PPO(env, size, number_of_state, number_of_action, max_epsisode=1,  result_path='./Result/', result_name='ppo')
    alg.train()