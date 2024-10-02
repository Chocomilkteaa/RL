import os
import cv2

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

class PolicyAndValueIteration:
    def __init__(self, env, size, number_of_state, number_of_action,
                 max_iter=10**6, tol=10**-3, discount_rate=0.99,
                 test_iter=10, result_path='./', result_name='video'):
        self.env = env
        self.size = size
        self.number_of_state = number_of_state
        self.number_of_action = number_of_action

        self.values = np.zeros(self.number_of_state)
        self.policies = np.array([self.env.action_space.sample() for _ in range(self.number_of_state)])

        self.rewards = np.zeros((self.number_of_state, self.number_of_action))
        self.transitions = np.zeros((self.number_of_state, self.number_of_action, self.number_of_state))

        self.max_iter = max_iter
        self.tol = tol
        self.discount_rate = discount_rate

        self.test_iter = test_iter

        self.result_path = result_path
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        
        self.result_name = result_name

    def getRewardsAndTransitions(self):
        for state in range(self.number_of_state):
            for action in range(self.number_of_action):
                for transition_info in self.env.unwrapped.P[state][action]:
                    transition, next_state, reward, done = transition_info
                    self.rewards[state, action] = reward
                    self.transitions[state, action, next_state] = transition
    
    def optimizePolicy(self):
        for i in range(self.max_iter):
            values_old = self.values.copy()

            self.values = np.max(self.rewards + np.matmul(self.transitions, self.discount_rate * values_old), axis=1)

            if np.max(np.abs(self.values - values_old)) < self.tol:
                break

        self.policies = np.argmax(self.rewards + np.matmul(self.transitions, self.discount_rate * self.values), axis=1)

    def getAction(self, state):
        return self.policies[state]

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

    def value_iteration(self):
        self.getRewardsAndTransitions()

        self.optimizePolicy()

        self.test()

    def policy_iteration(self):
        self.getRewardsAndTransitions()

        for i in range(self.test_iter):
            policies_old = self.policies.copy()

            self.optimizePolicy()

            if np.all(np.equal(self.policies, policies_old)):
                break

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

    print('Value Iteration:')
    vi = PolicyAndValueIteration(env, size, number_of_state, number_of_action, result_path='./Result/', result_name='vi')
    vi.value_iteration()

    print('Policy Iteration:')
    pi = PolicyAndValueIteration(env, size, number_of_state, number_of_action, result_path='./Result/', result_name='pi')
    pi.policy_iteration()
