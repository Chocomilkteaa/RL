import os
import cv2

import numpy as np
import gymnasium as gym

class QLearning:
    def __init__(self, env, size,
                 max_iter=10**4, tol=10**-3, 
                 explore_rate=0.99, learning_rate=0.5, discount_rate=0.99, 
                 test_iter=10, result_path='./', result_name='video'):
        self.env = env
        self.size = size
        self.number_of_state = self.env.observation_space.n
        self.number_of_action = self.env.action_space.n

        self.q_table = np.zeros((self.number_of_state, self.number_of_action))

        self.max_iter = max_iter
        self.tol = tol

        self.explore_rate = explore_rate
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate

        self.test_iter = test_iter

        self.result_path = result_path
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        
        self.result_name = result_name

    def getAction(self, state):
        return np.argmax(self.q_table[state])

    def train(self):
        for i in range(self.max_iter):
            state, info = self.env.reset()

            self.explore_rate *= 0.9

            while True:
                random_num = np.random.uniform(0,1)

                if random_num > self.explore_rate:
                    action = self.getAction(state)
                else:
                    action = self.env.action_space.sample()

                next_state, reward, terminated, truncated, info = self.env.step(action)

                self.q_table[state, action] = self.q_table[state, action] + \
                    self.learning_rate * (reward + self.discount_rate * np.max(self.q_table[next_state, :]) - self.q_table[state, action])
                
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
        
        state, info = self.env.reset(seed=np.argmax(trajectory_rewards))

        writer = cv2.VideoWriter(os.path.join(self.result_path, f'{self.result_name}.avi'),cv2.VideoWriter_fourcc(*'DIVX'), 2, self.size)

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

if __name__ == '__main__':
    np.random.seed(0)
    
    env = gym.make("Taxi-v3", render_mode='rgb_array')
    observation, info = env.reset()
    size = (env.render().shape[1], env.render().shape[0])

    ql = QLearning(env, size, result_path='./Result/', result_name='vi')
    ql.qlearning()