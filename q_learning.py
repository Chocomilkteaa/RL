import os
import cv2

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

class QLearning:
    def __init__(self, env, size, number_of_state, number_of_action,
                 max_iter=10**4, tol=10**-3, 
                 explore_rate=0.99, explore_rate_decay=0.9, learning_rate=0.5, discount_rate=0.99, 
                 test_iter=10, result_path='./', result_name='video', frame_rate=30):
        self.env = env
        self.size = size
        self.number_of_state = number_of_state
        self.number_of_action = number_of_action

        self.q_table = np.zeros((self.number_of_state, self.number_of_action))

        self.max_iter = max_iter
        self.tol = tol

        self.explore_rate = explore_rate
        self.explore_rate_decay = explore_rate_decay

        self.learning_rate = learning_rate
        self.discount_rate = discount_rate

        self.test_iter = test_iter

        self.result_path = result_path
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        
        self.result_name = result_name

        self.frame_rate = frame_rate

    def updateExploreRate(self):
        self.explore_rate *= self.explore_rate_decay

    def getAction(self, state):
        return np.argmax(self.q_table[state])
    
    def getActionRandom(self, state):
        random_num = np.random.uniform(0,1)

        if random_num > self.explore_rate:
            action = self.getAction(state)
        else:
            action = self.env.action_space.sample()

        return action
    
    def updateQTable(self, state, action, reward, next_state):
        self.q_table[state, action] = self.q_table[state, action] + \
                    self.learning_rate * (reward + self.discount_rate * np.max(self.q_table[next_state, :]) - self.q_table[state, action])

    def train(self):
        for i in range(self.max_iter):
            state, info = self.env.reset(seed=i)

            self.updateExploreRate()

            while True:
                action = self.getActionRandom(state)

                next_state, reward, terminated, truncated, info = self.env.step(action)

                self.updateQTable(state, action, reward, next_state)
                
                if terminated or truncated:
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

    def qlearning(self):
        self.train()

        self.test()

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
    np.random.seed(0)
    
    env = gym.make("Taxi-v3", render_mode='rgb_array')
    
    size, number_of_state, number_of_action = getEnvInfo(env)

    alg = QLearning(env, size, number_of_state, number_of_action, result_path='./Result/', result_name='ql', frame_rate=2)
    alg.qlearning()