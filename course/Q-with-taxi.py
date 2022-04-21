import numpy as np
import random
import gym
import time

env = gym.make("Taxi-v3")
# +---------+
# |R: | : :G|
# | : | : : |
# | : : : : |
# | | : | : |
# |Y| : |B: |
# +---------+

state_space = env.observation_space.n
# There are  500 possible states

action_space = env.action_space.n
# There are  6 possible actions

Q = np.zeros((state_space, action_space))
# (500, 6)

total_episodes = 25000  # Total number of training episodes
total_test_episodes = 100  # Total number of test episodes
max_steps = 200  # Max steps per episode

learning_rate = 0.01  # Learning rate
gamma = 0.09  # Discounting factor

epsilon = 1.0  # Exploration rate
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.001  # Minimum exploration probability
decay_rate = 0.01  # Exponential decay rate for exploration prob


def epsilon_greedy_policy(Q, state):
    if (random.uniform(0, 1) > epsilon):
        action = np.argmax(Q[state])
    else:
        action = env.action_space.sample()

    return action


for episode in range(total_episodes):
    state = env.reset()
    step = 0
    done = False

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    for step in range(max_steps):
        action = epsilon_greedy_policy(Q, state)
        new_state, reward, done, info = env.step(action)

        Q[state][action] = Q[state][action] + learning_rate * (reward + gamma * np.max(Q[new_state]) - Q[state][action])

        if done:
            break

        state = new_state

rewards = []
frames = []
for episode in range(total_test_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    print('-' * 20)
    print('episode ', episode)
    for step in range(max_steps):
        env.render()
        action = np.argmax(Q[state][:])
        new_state, reward, done, info = env.step(action)
        total_rewards += reward

        if done:
            rewards.append(total_rewards)
            break
        state = new_state

env.close()
print("Score over time: " + str(sum(rewards)/total_test_episodes))