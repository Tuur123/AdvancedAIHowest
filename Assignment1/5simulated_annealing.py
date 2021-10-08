import gym
import matplotlib.pyplot as plt
from numpy.lib.function_base import average
from scipy.stats.stats import obrientransform
import seaborn as sns
import numpy as np

env = gym.make('CartPole-v0')

average_reward = []

previous_weights = (2 * np.random.rand(4) - 1.0)

best_reward = 0
previous_reward = 0

temperature = 100000
cooling_rate = 0.993
spread = 0.001

def computeAction(state, weights):
    p = np.matmul(weights, state)
    if p < 0:
        action = 0
    else:       
        action = 1
    return action


for iteration in range(1000):

    current_reward = 0

    for episode in range(20):

        current_weights = previous_weights + np.random.normal(loc=0, scale = spread)
        observation = env.reset()
        for t in range(200):
        
            observation, reward, done, info = env.step(computeAction(observation, current_weights))
            current_reward = current_reward + reward / 20
            if done:
                break

    average_reward.append(current_reward)

    if current_reward > best_reward:
        best_reward = current_reward
        best_weights = current_weights

    else:        
        reward_difference = (current_reward - previous_reward)
        p = np.exp(reward_difference/temperature)
        if  np.random.rand() < p:
            previous_reward = current_reward
            previous_weights = current_weights

    temperature = cooling_rate * temperature

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

sns.histplot(average_reward)
plt.show()