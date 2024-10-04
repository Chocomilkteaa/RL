import os
import cv2

from collections import namedtuple, deque
import random
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

import torch

class Policy(torch.nn.Module):
    def __init__(self, number_of_state, number_of_action, hidden_size, device):
        super(Policy, self).__init__()
        self.layer1 = torch.nn.Linear(number_of_state, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, number_of_action)

        self.device = device

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x = torch.nn.functional.softmax(self.layer3(x), dim=1)
        return x
    
    def getActionAndProb(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)

        action = dist.sample()

        return action.item(), dist.log_prob(action)
    
class Reinforce(object):
    def __init__(self, env, size, number_of_state, number_of_action, 
                hidden_size=128, max_epsisode=1000, learning_rate=1e-4, discount_rate=0.99,
                target_score=500, test_iter=10, result_path='./Result', result_name='video', frame_rate=30):
        
        self.env = env
        self.size = size
        self.number_of_state = number_of_state
        self.number_of_action = number_of_action

        self.max_epsisode = max_epsisode
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate

        self.target_score = target_score

        self.test_iter = test_iter

        self.result_path = result_path
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        
        self.result_name = result_name

        self.frame_rate = frame_rate

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.policy_net = Policy(number_of_state, number_of_action, hidden_size, self.device).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        self.eps = np.finfo(np.float32).eps.item()

    def optimizePolicy(self, params):
        R = 0
        returns = deque()
        for r in reversed(params['rewards']):
            R = r + R * self.discount_rate
            returns.appendleft(R)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        loss = torch.cat([-a * b for a, b in zip(params['log_probs'], returns)]).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def getActionAndProb(self, state):
        return self.actor_net.getActionAndProb(state)
    
    def getAction(self, state):
        action, log_prob = self.actor_net.getActionAndProb(state)
        return action

    def train(self):
        average_trajectory_reward = deque(maxlen=100)

        for episode in range(self.max_epsisode):
            state, info = self.env.reset(seed = episode)

            rewards = []
            log_probs = []

            while True:
                action, log_prob = self.getActionAndProb(state)

                log_probs.append(log_prob)

                next_state, reward, terminated, truncated, info = env.step(action)

                rewards.append(reward)

                if terminated or truncated:
                    print(f'Episode: {episode} ends with reward {sum(rewards)}')
                    average_trajectory_reward.append(sum(rewards))
                    break

                state = next_state

            if np.mean(average_trajectory_reward) >= self.target_score:
                print(f'solved with {episode} epochs')
                break

            params = {'rewards': rewards, 'log_probs': log_probs}

            self.optimizePolicy(params)

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
            action, _ = self.getAction(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            state = next_state

            writer.write(self.env.render())

            if terminated or truncated:
                writer.release()

                break

    def reinforce(self):
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

    alg = Reinforce(env, size, number_of_state, number_of_action, 
                    learning_rate=2.5*1e-4, max_epsisode=2000, 
                    result_path='./Result/', result_name='reinforce')
    alg.reinforce()