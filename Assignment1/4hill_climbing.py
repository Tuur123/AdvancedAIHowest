import gym
import matplotlib.pyplot as plt
from numpy.lib.function_base import average
from scipy.stats.stats import obrientransform
import seaborn as sns
import numpy as np

env = gym.make('CartPole-v0')

average_reward = []

noise_scaling = 1e-3
best_weights = (2 * np.random.rand(4) - 1.0)
best_reward = 0

def computeAction(state, weights):
    p = np.matmul(weights, state)
    if p < 0:
        action = 0
    else:       
        action = 1
    return action

for iteration in range(1000):

    total_reward = 0

    for episode in range(20):

        weights = best_weights + noise_scaling * (2 * np.random.rand(4) - 1.0)
        observation = env.reset()
        for t in range(200):
        
            observation, reward, done, info = env.step(computeAction(observation, weights))
            total_reward += reward
            if done:
                break

    average_reward.append(total_reward / 20)

    if total_reward > best_reward:
        best_reward = total_reward
        best_weights = weights

print(f"Best reward: {np.max(average_reward)} with weights: {best_weights}")

total_reward = 0
for episode in range(1000):
    observation = env.reset()
    for t in range(200):
     
        observation, reward, done, info = env.step(computeAction(observation, best_weights))
        total_reward += reward
        if done:
            break


print(f"Average reward over 1000 episodes with best weights: {total_reward / 1000}")
env.close()

sns.distplot(average_reward)
plt.show()